import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict
import logging
import traceback
import numpy as np

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchedulingSolver:
    """
    A generic solver that interprets Constraint Definition Language (CDL)
    using Google OR-Tools CP-SAT.
    """
    def __init__(self, pod_data, plan_json):
        self.raw_pod_data = pod_data
        self.pod_data_df = pd.DataFrame(pod_data)

        # --- Preprocessing ---
        # Convert numeric columns safely
        for col in self.pod_data_df.columns:
            # Force numeric conversion where possible
            try:
                self.pod_data_df[col] = pd.to_numeric(self.pod_data_df[col], errors='ignore')
            except Exception as e:
                pass

        # Ensure IDCS_MONTH is numeric if present
        if 'IDCS_MONTH' in self.pod_data_df.columns:
             self.pod_data_df['IDCS_MONTH'] = pd.to_numeric(self.pod_data_df['IDCS_MONTH'], errors='coerce').fillna(0).astype(int)

        self.pod_data = self.pod_data_df.to_dict('records')
        self.global_params = plan_json.get('global_parameters', {})
        self.constraints = plan_json.get('constraints', [])

        self.model = cp_model.CpModel()
        self.pod_vars = {} # {pod_name: IntVar(AssignedMonth)}
        self.pod_map = {str(pod.get('PODNAME')): pod for pod in self.pod_data if pod.get('PODNAME')}

        # Cache for indicator variables: {(pod_name, month): BoolVar}
        self.pod_month_indicators = {}

        # Objective Terms (Values to Minimize)
        self.objective_terms = []

    def _get_horizon(self):
        # Default horizon: 12 to 48 months
        total_months = int(self.global_params.get("total_duration_months", 48))
        start_month = int(self.global_params.get("start_month_index", 1))
        return start_month, max(start_month, total_months)

    def solve(self):
        logger.info("Engine: Initializing variables...")
        start_month, max_month = self._get_horizon()

        # 1. Create Decision Variables (AssignedMonth)
        for pod in self.pod_data:
            pod_name = str(pod.get('PODNAME'))
            if pod_name:
                # Variable: AssignedMonth for this pod
                v = self.model.NewIntVar(start_month, max_month, f"month_{pod_name}")
                self.pod_vars[pod_name] = v

                # Create boolean indicators for each month (needed for capacity constraints)
                for m in range(start_month, max_month + 1):
                    b = self.model.NewBoolVar(f"ind_{pod_name}_{m}")
                    # b <=> (v == m)
                    self.model.Add(v == m).OnlyEnforceIf(b)
                    self.model.Add(v != m).OnlyEnforceIf(b.Not())
                    self.pod_month_indicators[(pod_name, m)] = b

        # 2. Apply Constraints (Universal Interpreter)
        logger.info(f"Engine: Applying {len(self.constraints)} Universal CDL Constraints...")
        for i, c in enumerate(self.constraints):
            try:
                logger.info(f"Applying Constraint {i+1}: {c.get('rule_type', 'Unknown')}")
                self._apply_universal_constraint(c)
            except Exception as e:
                logger.error(f"ERROR applying constraint {c.get('rule_type')}: {e}")
                traceback.print_exc()

        # 3. Objective: Minimize Makespan + Custom Terms
        # Minimize the maximum month used (Default Objective)
        makespan = self.model.NewIntVar(start_month, max_month, 'makespan')
        self.model.AddMaxEquality(makespan, list(self.pod_vars.values()))

        # Combine Default + Custom Objectives
        # We weigh Makespan heavily (500) as per user preference, but let custom terms influence distribution.
        total_objective = (makespan * 500)

        if self.objective_terms:
            logger.info(f"Engine: Adding {len(self.objective_terms)} custom objective terms.")
            total_objective += sum(self.objective_terms)

        self.model.Minimize(total_objective)

        # 4. Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.info(f"Solution Found: {solver.StatusName(status)}")
            return self._extract_schedule(solver)
        else:
            logger.warning(f"No Solution: {solver.StatusName(status)}")
            return None

    # ==============================================================================
    # UNIVERSAL CONSTRAINT INTERPRETER
    # ==============================================================================

    def _apply_universal_constraint(self, cdl):
        params = cdl.get('params', {})
        rule_type = params.get('type', 'CONSTRAINT') # CONSTRAINT or OBJECTIVE
        weight = params.get('weight', 1)
        action = params.get('action', 'MINIMIZE')

        # 1. Iterator Logic (e.g. For Each Month)
        iterator = params.get('iterator')
        if iterator:
            var_name = iterator.get('variable', 'i')
            rng = iterator.get('range', [1, 1])

            if isinstance(rng, str) and rng == 'ALL_MONTHS':
                start, end = self._get_horizon()
            elif isinstance(rng, list) and len(rng) == 2:
                start, end = rng
            else:
                start, end = 1, 1 # Fallback

            for i in range(start, end + 1):
                self._apply_inner(
                    params.get('scope', {'type': 'GLOBAL'}),
                    params.get('expression', {}),
                    {var_name: i},
                    rule_type, weight, action
                )
        else:
            self._apply_inner(
                params.get('scope', {'type': 'GLOBAL'}),
                params.get('expression', {}),
                {},
                rule_type, weight, action
            )

    def _apply_inner(self, scope, expr, extra_vars, rule_type='CONSTRAINT', weight=1, action='MINIMIZE'):
        contexts = self._resolve_scope(scope)
        for ctx in contexts:
            # Merge extra_vars into context vars
            ctx['vars'].update(extra_vars)

            # Evaluate the expression
            # For CONSTRAINT: returns boolean (True/False/BoolVar)
            # For OBJECTIVE: returns numeric value (Int/IntVar/LinearExpr) to minimize
            res = self._eval(expr, ctx)

            if rule_type == 'OBJECTIVE':
                # Add to objective terms
                if res is None: continue

                # Handle List result (if scope returned multiple items in one context)
                terms = res if isinstance(res, list) else [res]

                for t in terms:
                    # Convert to LinearExpr/IntVar if constant
                    term = self._to_int_var(t)

                    # Apply Weight
                    weighted_term = term * weight

                    # Apply Action (Minimize is default +)
                    if action == 'MAXIMIZE':
                        # To Maximize X, we Minimize -X
                        weighted_term = weighted_term * -1

                    self.objective_terms.append(weighted_term)

            else:
                # Standard Constraint
                if hasattr(res, 'Index'):
                    self.model.Add(res == 1)
                elif isinstance(res, bool):
                    if not res:
                        self.model.Add(self.model.NewConstant(0) == 1)

    def _resolve_scope(self, scope):
        sType = scope.get('type', 'GLOBAL')
        filter_expr = scope.get('filter')

        if sType == 'GLOBAL':
            target_pods = list(self.pod_vars.keys())
        elif sType == 'FOR_EACH_UNIQUE_COMBINATION':
            cols = scope.get('columns', [])
            # ... (Grouping logic not fully used if we filter later, but let's stick to current)
            # Actually, standard scope resolution returns ONE context per group.
            # If we want per-pod filtering, we usually iterate groups.
            # But wait, OBJECTIVES often apply PER POD (e.g. Prioritize Small Families).
            # If scope is GLOBAL, we get ALL pods.
            # If we want to filter specific pods, we need `scope: {type: GLOBAL, filter: ...}`.
            # My current _resolve_scope implementation ignores `filter`.
            # I should implement filtering in _resolve_scope.
            target_pods = list(self.pod_vars.keys())

            # Logic for grouping
            cols = scope.get('columns', [])
            groups = defaultdict(list)

            # Pre-filter pods if needed?
            # Or filter AFTER grouping.

            for pname in target_pods:
                # Apply filter if exists
                if filter_expr:
                     # Check filter. Need to evaluate expression against pod data.
                     # Using _eval requires context.
                     # This is circular if _eval calls scope.
                     # But filter expr should be simple static check.
                     # Let's verify filter for this pod.
                     sub_ctx = {'pods': [pname], 'vars': {}}
                     # We can't use self._eval easily here if it calls other things?
                     # Actually _eval is robust.
                     if not self._check_filter_static(filter_expr, sub_ctx):
                         continue

                key = tuple(str(self.pod_map[pname].get(c, 'N/A')) for c in cols)
                groups[key].append(pname)

            return [{'pods': pods, 'vars': {'group_key': k}} for k, pods in groups.items()]

        # GLOBAL with Filter
        if sType == 'GLOBAL':
             filtered_pods = []
             for pname in list(self.pod_vars.keys()):
                 if filter_expr:
                     sub_ctx = {'pods': [pname], 'vars': {}}
                     if not self._check_filter_static(filter_expr, sub_ctx):
                         continue
                 filtered_pods.append(pname)
             return [{'pods': filtered_pods, 'vars': {}}]

        return []

    def _check_filter_static(self, expr, context):
        """Helper to evaluate static filter expressions on pod data."""
        # Use a separate static evaluation that returns python bool/int or None
        res = self._eval_static(expr, context)
        if isinstance(res, bool): return res
        if res is None: return True # Dynamic/Unknown -> Include
        return bool(res)

    def _eval_static(self, node, context):
        """Evaluate expression statically. Returns None if dynamic variables involved."""
        if isinstance(node, (int, float, str, bool)): return node
        if node is None: return None
        if isinstance(node, list):
            res_list = [self._eval_static(x, context) for x in node]
            if any(r is None for r in res_list): return None
            return res_list

        if not isinstance(node, dict): return None

        if 'variable' in node:
            var = node['variable']
            # If variable is in context['vars'] (iterators), return value
            if var in context.get('vars', {}): return context['vars'][var]

            # If variable is column data (static for a pod context)
            if 'pods' in context and len(context['pods']) == 1:
                 p = context['pods'][0]
                 val = self.pod_map[p].get(var)
                 if val is not None: return val

            # If variable is 'AssignedMonth' (dynamic), return None
            return None

        if 'operator' in node:
            op = node['operator']
            operands = node.get('operands', [])
            left = self._eval_static(operands[0], context)
            right = self._eval_static(operands[1], context) if len(operands) > 1 else None

            if left is None or (len(operands) > 1 and right is None): return None

            # Perform static op
            try:
                if op == '==': return left == right
                if op == '!=': return left != right
                if op == '<': return left < right
                if op == '<=': return left <= right
                if op == '>': return left > right
                if op == '>=': return left >= right
                if op == 'IN':
                    if isinstance(right, list): return left in right
                    return False
                if op == 'AND': return bool(left) and bool(right) # operands[0] and operands[1]
                # Fix: AND operands is list?
                # _eval_operator passes list to logic gate?
                # Logic gates take list of operands?
                # My _eval_operator for AND calls _apply_logic_gate which takes list.
                # Here we simplify: if AND, evaluate all operands.
                pass
            except: return None

            # Logic Ops (List operands)
            if op == 'AND':
                # Re-eval as list
                # operands is list of nodes.
                vals = [self._eval_static(x, context) for x in operands]
                if any(v is None for v in vals): return None
                return all(vals)
            if op == 'OR':
                vals = [self._eval_static(x, context) for x in operands]
                if any(v is None for v in vals): return None
                return any(vals)
            if op == 'NOT':
                return not left

        return None

    def _eval(self, node, context):
        if isinstance(node, (int, float, str, bool)): return node
        if node is None: return None
        if isinstance(node, list): return [self._eval(x, context) for x in node]

        if not isinstance(node, dict): return node

        if 'variable' in node:
            return self._eval_variable(node, context)

        if 'function' in node:
            return self._eval_function(node, context)

        if 'operator' in node:
            return self._eval_operator(node, context)

        return None

    def _eval_variable(self, node, context):
        var = node['variable']
        if var == 'AssignedMonth':
            return [self.pod_vars[p] for p in context['pods']]
        if var in context.get('vars', {}):
            return context['vars'][var]

        # Return raw values. Try to ensure they are numeric if they look like it?
        # Pandas should have handled it, but let's be safe.
        raw_vals = [self.pod_map[p].get(var) for p in context['pods']]
        return raw_vals

    def _eval_function(self, node, context):
        func = node['function']
        args = node.get('arguments', [])

        if func in ['GET_SUM_FOR_MONTH', 'GET_POD_COUNT_FOR_MONTH']:
            return self._eval_capacity_function(func, args, context)

        if func == 'GET_POD_COUNT_FOR_PERIOD':
            return self._eval_period_capacity_function(args, context)

        if func == 'ALL_MEMBERS_HAVE_SAME_VALUE':
            return self._eval_all_same(args, context)

        if func == 'SUM':
            # Sum of list
            # Flatten if args[0] resulted in a list of lists?
            vals = self._eval(args[0], context)

            # Ensure vals is a flat list of numbers
            if isinstance(vals, list):
                # Filter out None or non-summable?
                # Or assume they are summable.
                # If they are strings, we try to float them.
                clean_vals = []
                for v in vals:
                    if isinstance(v, list): continue # Should not happen unless nested vars
                    try:
                        clean_vals.append(float(v))
                    except:
                        # If variable (IntVar), keep it
                        if hasattr(v, 'Index'):
                            clean_vals.append(v)
                        else:
                            clean_vals.append(0) # Default for bad data?

                return sum(clean_vals)
            return vals

        if func == 'COUNT':
            vals = self._eval(args[0], context)
            if isinstance(vals, list): return len(vals)
            return 1

        return None

    def _eval_period_capacity_function(self, args, context):
        # args: [StartMonth, EndMonth, OptionalFilter]
        start_m = self._eval(args[0], context)
        end_m = self._eval(args[1], context)
        filter_node = args[2] if len(args) > 2 else None

        if not isinstance(start_m, int) or not isinstance(end_m, int):
            return 0 # Dynamic period not supported yet

        sum_terms = []
        target_pods = context['pods']

        # Optimization: Pre-calculate filter for all pods once?
        # But filter might be dynamic.
        # If filter is static, we can prune pods list.

        relevant_pods = target_pods
        if filter_node:
             # Try static filtering first
             static_filtered = []
             for p in target_pods:
                 if self._check_filter_static(filter_node, {'pods': [p], 'vars': {}}):
                     static_filtered.append(p)
             relevant_pods = static_filtered

        for p in relevant_pods:
            # We want to check if AssignedMonth(p) is in [start_m, end_m].
            # Variable v.
            # v >= start AND v <= end.
            # Indicator?
            # We have indicators for each month.
            # Sum( indicators[p][m] for m in start..end )

            # Check filter (dynamic)
            condition_vars = []
            if filter_node:
                # If static check passed, we still need to check dynamic?
                # If static check returned True (maybe dynamic), we must check dynamic expression.
                # If static check returned False, we pruned it.
                # Re-eval filter dynamically
                res = self._eval(filter_node, {'pods': [p], 'vars': {}})
                if hasattr(res, 'Index'):
                    condition_vars.append(res)
                elif isinstance(res, bool) and not res:
                    continue # Should have been pruned, but just in case

            # Period Indicators
            pod_period_indicators = []
            for m in range(start_m, end_m + 1):
                ind = self.pod_month_indicators.get((p, m))
                if ind is not None:
                    pod_period_indicators.append(ind)

            if not pod_period_indicators: continue

            # If pod is in period => Sum(indicators) == 1 (since months are mutually exclusive)
            # So `is_in_period` = Sum(indicators).
            # If we have filter, we need `is_in_period AND Filter`.

            in_period_var = sum(pod_period_indicators) # This is an LinearExpr (0 or 1)

            if condition_vars:
                # We need (InPeriod AND Filter).
                # InPeriod is 0/1. Filter is BoolVar.
                # Product: InPeriod * Filter.
                # LinearExpr * BoolVar -> Allowed?
                # Yes, CP-SAT allows Sum(Vars).

                # If multiple filter conditions, AND them first
                if len(condition_vars) > 1:
                    filter_combined = self.model.NewBoolVar(f'filter_comb_{p}')
                    self.model.AddBoolAnd(condition_vars).OnlyEnforceIf(filter_combined)
                    self.model.AddBoolOr([v.Not() for v in condition_vars]).OnlyEnforceIf(filter_combined.Not())
                    final_filter = filter_combined
                else:
                    final_filter = condition_vars[0]

                # Term: in_period_var * final_filter
                # Since in_period_var is sum of booleans (mutually exclusive), it is 0 or 1.
                # We can treat it as a variable?
                # But it's an expression.
                # We can use AddMultiplicationEquality if we bind in_period to a var?
                # Or just use `OnlyEnforceIf`.

                # Actually, simply: Sum( ind_m * filter ) for all m.
                # For each month m: term_m = ind_m AND filter.
                # Total = Sum(term_m).

                for m in range(start_m, end_m + 1):
                    ind = self.pod_month_indicators.get((p, m))
                    if ind is not None:
                        # term = ind AND final_filter
                        term = self.model.NewBoolVar(f'term_{p}_{m}')
                        self.model.AddBoolAnd([ind, final_filter]).OnlyEnforceIf(term)
                        self.model.AddBoolOr([ind.Not(), final_filter.Not()]).OnlyEnforceIf(term.Not())
                        sum_terms.append(term)
            else:
                # No filter, just add period indicators
                sum_terms.extend(pod_period_indicators)

        return sum(sum_terms)

    def _eval_capacity_function(self, func, args, context):
        m_idx = self._eval(args[0], context)
        col_name = args[1]
        filter_node = args[2] if len(args) > 2 else None

        target_pods = context['pods']
        sum_terms = []

        for p in target_pods:
            condition_vars = []

            if filter_node:
                sub_ctx = {'pods': [p], 'vars': {}}
                res = self._eval(filter_node, sub_ctx)
                if isinstance(res, bool):
                    if not res: continue
                elif hasattr(res, 'Index'):
                    condition_vars.append(res)
                else:
                    # If res is None or unknown type, skip
                    continue

            if isinstance(m_idx, int):
                ind = self.pod_month_indicators.get((p, m_idx))
                if ind is not None:
                    condition_vars.append(ind)
                else:
                    continue
            else:
                logger.warning("Variable month index in capacity constraint not supported yet.")
                continue

            if not condition_vars:
                continue

            if len(condition_vars) == 1:
                active_var = condition_vars[0]
            else:
                active_var = self.model.NewBoolVar(f"active_{p}_{col_name}_{m_idx}")
                self.model.AddBoolAnd(condition_vars).OnlyEnforceIf(active_var)
                self.model.AddBoolOr([v.Not() for v in condition_vars]).OnlyEnforceIf(active_var.Not())

            if func == 'GET_POD_COUNT_FOR_MONTH':
                val = 1
            else:
                val = self.pod_map[p].get(col_name, 0)
                try:
                    # CP-SAT only supports integers.
                    # We cast float to int. (Rounding or Truncating).
                    # For now, standard int() cast.
                    val = int(float(val))
                except: val = 0

            if val == 0: continue
            sum_terms.append(active_var * val)

        return sum(sum_terms)

    def _eval_all_same(self, args, context):
        target_var = args[0] if len(args) > 0 else "AssignedMonth"
        values = self._eval({'variable': target_var}, context)

        if not values or len(values) < 2:
            return self.model.NewConstant(1)

        if not hasattr(values[0], 'Index'):
            first = values[0]
            return self.model.NewConstant(1 if all(v == first for v in values) else 0)

        all_same = self.model.NewBoolVar('all_same')
        first = values[0]
        bools = []
        for other in values[1:]:
            b = self.model.NewBoolVar(f'eq_{first.Name()}_{other.Name()}')
            self.model.Add(first == other).OnlyEnforceIf(b)
            self.model.Add(first != other).OnlyEnforceIf(b.Not())
            bools.append(b)

        self.model.AddBoolAnd(bools).OnlyEnforceIf(all_same)
        self.model.AddBoolOr([b.Not() for b in bools]).OnlyEnforceIf(all_same.Not())

        return all_same

    def _eval_operator(self, node, context):
        op = node['operator']
        operands = node.get('operands', [])

        left_raw = self._eval(operands[0], context)
        right_raw = self._eval(operands[1], context) if len(operands) > 1 else None

        if op in ['==', '!=', '<', '<=', '>', '>=']:
            return self._apply_comparison(op, left_raw, right_raw)

        if op == 'IN':
            return self._apply_in(left_raw, right_raw)

        if op in ['+', '-', '*', '/', '%']:
            return self._apply_arithmetic(op, left_raw, right_raw)

        if op == 'AND':
            ops = [self._eval(x, context) for x in operands]
            return self._apply_logic_gate('AND', ops)

        if op == 'OR':
            ops = [self._eval(x, context) for x in operands]
            return self._apply_logic_gate('OR', ops)

        if op == 'NOT':
            val = self._eval(operands[0], context)
            return self._apply_not(val)

        if op == 'IMPLIES':
            return self._apply_implies(left_raw, right_raw)

        return None

    def _apply_comparison(self, op, left, right):
        if isinstance(left, list) and not isinstance(right, list):
            bools = [self._apply_scalar_comparison(op, l, right) for l in left]
            return self._apply_logic_gate('AND', bools)

        if not isinstance(left, list) and isinstance(right, list):
            bools = [self._apply_scalar_comparison(op, left, r) for r in right]
            return self._apply_logic_gate('AND', bools)

        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right): return self.model.NewConstant(0)
            bools = [self._apply_scalar_comparison(op, l, r) for l, r in zip(left, right)]
            return self._apply_logic_gate('AND', bools)

        return self._apply_scalar_comparison(op, left, right)

    def _apply_arithmetic(self, op, left, right):
        # 1. Handle Broadcasting (List inputs)
        if isinstance(left, list) and not isinstance(right, list):
            return [self._apply_arithmetic(op, l, right) for l in left]
        if not isinstance(left, list) and isinstance(right, list):
            return [self._apply_arithmetic(op, left, r) for r in right]
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right): return None # Error or Mismatch
            return [self._apply_arithmetic(op, l, r) for l, r in zip(left, right)]

        # 2. Scalar Logic
        # Handle constants (Python Math)
        is_left_static = isinstance(left, (int, float))
        is_right_static = isinstance(right, (int, float))

        if is_left_static and is_right_static:
            if op == '+': return left + right
            if op == '-': return left - right
            if op == '*': return left * right
            if op == '/': return left / right if right != 0 else 0
            if op == '%': return left % right if right != 0 else 0

        # Handle Linear Expressions (CP-SAT)

        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*':
            if not is_left_static and not is_right_static:
                left_var = self._to_int_var(left)
                right_var = self._to_int_var(right)
                prod = self.model.NewIntVar(0, 10000000, f'prod_{id(left)}_{id(right)}')
                self.model.AddMultiplicationEquality(prod, [left_var, right_var])
                return prod
            return left * right

        if op == '/':
            if not is_left_static or not is_right_static:
                left_var = self._to_int_var(left)
                right_var = self._to_int_var(right)
                div = self.model.NewIntVar(0, 1000000, f'div_{id(left)}')
                self.model.AddDivisionEquality(div, left_var, right_var)
                return div
            return int(left / right)

        if op == '%':
            if not is_left_static or not is_right_static:
                left_var = self._to_int_var(left)
                right_var = self._to_int_var(right)
                mod_res = self.model.NewIntVar(0, 1000000, f'mod_{id(left)}')
                self.model.AddModuloEquality(mod_res, left_var, right_var)
                return mod_res
            return int(left % right)

        return None

    def _to_int_var(self, expr):
        """Helper to convert a LinearExpr (or int) to a single IntVar."""
        if isinstance(expr, int): return self.model.NewConstant(expr)
        if hasattr(expr, 'Index'): return expr # Already IntVar
        # It's a LinearExpr (e.g. x + 5)
        # We need to bind it to a new variable: v = expr
        # Bounds? LinearExpr can be anything.
        # We assume reasonable bounds for planning (0..1000 or -1000..1000)
        # AssignedMonth is 0..48. DB_SIZE can be large.
        # Safe bounds: -1M to 1M?
        # If it's DB_SIZE, it's huge.
        # But usually we don't do DB_SIZE % X. We do Month % 3.
        v = self.model.NewIntVar(-1000000, 1000000, f'expr_{id(expr)}')
        self.model.Add(v == expr)
        return v

    def _apply_scalar_comparison(self, op, left, right):
        # Force numeric comparison if strings are passed but look like numbers
        if isinstance(left, str) and isinstance(right, (int, float)):
             try: left = float(left)
             except: pass
        if isinstance(right, str) and isinstance(left, (int, float)):
             try: right = float(right)
             except: pass

        # Check for primitive types (static evaluation)
        # Note: numpy types might need special handling, but usually behave like primitives
        is_left_static = isinstance(left, (int, float, str, bool)) or left is None
        is_right_static = isinstance(right, (int, float, str, bool)) or right is None

        if is_left_static and is_right_static:
            try:
                if op == '==': res = (left == right)
                elif op == '!=': res = (left != right)
                elif op == '<': res = (left < right)
                elif op == '<=': res = (left <= right)
                elif op == '>': res = (left > right)
                elif op == '>=': res = (left >= right)
                else: res = False
            except TypeError:
                # Comparison not supported (e.g. str < int), return False
                res = False
            return self.model.NewConstant(1 if res else 0)

        # Dynamic Evaluation (CP-SAT)
        b = self.model.NewBoolVar(f'cmp_{op}')
        if op == '==':
            self.model.Add(left == right).OnlyEnforceIf(b)
            self.model.Add(left != right).OnlyEnforceIf(b.Not())
        elif op == '!=':
            self.model.Add(left != right).OnlyEnforceIf(b)
            self.model.Add(left == right).OnlyEnforceIf(b.Not())
        elif op == '<':
            self.model.Add(left < right).OnlyEnforceIf(b)
            self.model.Add(left >= right).OnlyEnforceIf(b.Not())
        elif op == '<=':
            self.model.Add(left <= right).OnlyEnforceIf(b)
            self.model.Add(left > right).OnlyEnforceIf(b.Not())
        elif op == '>':
            self.model.Add(left > right).OnlyEnforceIf(b)
            self.model.Add(left <= right).OnlyEnforceIf(b.Not())
        elif op == '>=':
            self.model.Add(left >= right).OnlyEnforceIf(b)
            self.model.Add(left < right).OnlyEnforceIf(b.Not())

        return b

    def _apply_in(self, left, right_list):
        if not isinstance(right_list, list): return self.model.NewConstant(0)

        if isinstance(left, list):
            bools = [self._apply_in(l, right_list) for l in left]
            return self._apply_logic_gate('AND', bools)

        # If left is dynamic (IntVar or LinearExpr) and right_list is static constants
        is_left_static = isinstance(left, (int, float, str, bool)) or left is None

        if not is_left_static:
            # Check if right list is all static
            if all((isinstance(r, (int, float, str, bool)) or r is None) for r in right_list):
                # Use allowed assignments or simple OR
                bools = []
                for r in right_list:
                    bools.append(self._apply_scalar_comparison('==', left, r))
                return self._apply_logic_gate('OR', bools)

        # Fallback (Generic loop)
        bools = [self._apply_scalar_comparison('==', left, r) for r in right_list]
        return self._apply_logic_gate('OR', bools)

    def _apply_logic_gate(self, gate, bools):
        filtered_bools = []
        for b in bools:
            if not hasattr(b, 'Index'):
                if gate == 'AND' and not b: return self.model.NewConstant(0)
                if gate == 'OR' and b: return self.model.NewConstant(1)
            else:
                filtered_bools.append(b)

        if not filtered_bools:
            return self.model.NewConstant(1 if gate == 'AND' else 0)

        res = self.model.NewBoolVar(f'gate_{gate}')
        if gate == 'AND':
            self.model.AddBoolAnd(filtered_bools).OnlyEnforceIf(res)
            self.model.AddBoolOr([b.Not() for b in filtered_bools]).OnlyEnforceIf(res.Not())
        elif gate == 'OR':
            self.model.AddBoolOr(filtered_bools).OnlyEnforceIf(res)
            self.model.AddBoolAnd([b.Not() for b in filtered_bools]).OnlyEnforceIf(res.Not())

        return res

    def _apply_not(self, val):
        if hasattr(val, 'Index'):
            return val.Not()
        return self.model.NewConstant(0 if val else 1)

    def _apply_implies(self, left, right):
        not_a = self._apply_not(left)
        return self._apply_logic_gate('OR', [not_a, right])

    def _extract_schedule(self, solver):
        data = []
        for p, v in self.pod_vars.items():
            data.append({'PODNAME': p, 'AssignedMonth': solver.Value(v)})
        return pd.DataFrame(data)

if __name__ == "__main__":
    pass
