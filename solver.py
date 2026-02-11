import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict
import os
import sys
import traceback
import json

# --- CONFIGURATION ---
TEMP_KNOWLEDGE_FILE = "_temp_knowledge.json"
POD_DATA_FILE = "adbs_data3.csv"
PLAN_OUTPUT_FILE = "_temp_plan.csv"

class SchedulingSolver:
    def __init__(self, pod_data, plan_json):
        self.pod_data = pod_data
        self.pod_data_df = pd.DataFrame(pod_data)

        # --- Preprocessing & Defaults ---
        self.pod_data_df = self.pod_data_df.fillna("UNKNOWN")
        if 'IDCS_MONTH' in self.pod_data_df.columns:
             self.pod_data_df['IDCS_MONTH'] = pd.to_numeric(self.pod_data_df['IDCS_MONTH'], errors='coerce').fillna(0).astype(int)

        self.pod_data = self.pod_data_df.to_dict('records')
        self.global_params = plan_json.get('global_parameters', {})
        self.constraints = plan_json.get('constraints', [])

        self.model = cp_model.CpModel()
        self.pod_vars = {} # {pod_name: IntVar}
        self.pod_map = {pod.get('PODNAME'): pod for pod in self.pod_data if pod.get('PODNAME')}

        # Indicators Cache: {(pod_name, month): BoolVar}
        self.pod_month_indicators = {}

    def _get_horizon(self):
        total_months = self.global_params.get("total_duration_months", 48)
        start_month = self.global_params.get("start_month_index", 12)
        return start_month, max(start_month, total_months)

    def solve(self):
        print("Engine: Initializing variables...")
        start_month, max_month = self._get_horizon()

        # 1. Create Variables
        for pod in self.pod_data:
            pod_name = pod.get('PODNAME')
            if pod_name:
                v = self.model.NewIntVar(start_month, max_month, f"month_{pod_name}")
                self.pod_vars[pod_name] = v
                # Create indicators lazily or eagerly? Eagerly is safer for generic constraints.
                for m in range(start_month, max_month + 1):
                    b = self.model.NewBoolVar(f"i_{pod_name}_{m}")
                    self.model.Add(v == m).OnlyEnforceIf(b)
                    self.model.Add(v != m).OnlyEnforceIf(b.Not())
                    self.pod_month_indicators[(pod_name, m)] = b

        # 2. Apply Constraints (Universal Interpreter)
        print("Engine: Applying Universal CDL Constraints...")
        for c in self.constraints:
            try:
                self._apply_universal_constraint(c)
            except Exception as e:
                print(f"ERROR applying constraint {c.get('rule_type')}: {e}")
                # traceback.print_exc()

        # 3. Objective (Minimize Makespan default)
        makespan = self.model.NewIntVar(start_month, max_month, 'makespan')
        self.model.AddMaxEquality(makespan, list(self.pod_vars.values()))
        self.model.Minimize(makespan)

        # 4. Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"Solution Found: {solver.StatusName(status)}")
            return self._extract_schedule(solver)
        else:
            print(f"No Solution: {solver.StatusName(status)}")
            return None

    # ==============================================================================
    # UNIVERSAL CONSTRAINT INTERPRETER
    # ==============================================================================

    def _apply_universal_constraint(self, cdl):
        """
        Interprets a CDL object with 'scope' and 'expression'.
        """
        scope = cdl.get('params', {}).get('scope', {'type': 'GLOBAL'})
        expr = cdl.get('params', {}).get('expression', {})

        contexts = self._resolve_scope(scope)
        for ctx in contexts:
            res = self._eval(expr, ctx)
            # If result is a BoolVar, enforce it being True
            if hasattr(res, 'Index'):
                self.model.Add(res == 1)
            # If result is generic boolean True/False (python), check it?
            # Usually strict constraints return a BoolVar.

    def _resolve_scope(self, scope):
        sType = scope.get('type', 'GLOBAL')

        if sType == 'GLOBAL':
            return [{'pods': list(self.pod_vars.keys()), 'vars': {}}]

        elif sType == 'FOR_EACH_UNIQUE_COMBINATION':
            cols = scope.get('columns', [])
            groups = defaultdict(list)
            for pname in self.pod_vars.keys():
                key = tuple(str(self.pod_map[pname].get(c, 'N/A')) for c in cols)
                groups[key].append(pname)
            return [{'pods': pods, 'vars': {'group_key': k}} for k, pods in groups.items()]

        return []

    def _eval(self, node, context):
        if isinstance(node, (int, float, str, bool)): return node
        if isinstance(node, list): return [self._eval(x, context) for x in node]
        if not isinstance(node, dict): return node

        # Variable
        if 'variable' in node:
            var = node['variable']
            if var == 'AssignedMonth':
                # Return LIST of IntVars for current context pods
                return [self.pod_vars[p] for p in context['pods']]
            # Data Column
            return [self.pod_map[p].get(var) for p in context['pods']]

        # Function
        if 'function' in node:
            return self._eval_function(node, context)

        # Operator
        if 'operator' in node:
            return self._eval_operator(node, context)

        return None

    def _eval_function(self, node, context):
        func = node['function']
        # Eval arguments first? Depends on function (lazy eval might be needed for COUNT)
        # But for now, let's assume we eval them.
        # Wait, ALL_MEMBERS arguments are meta-data sometimes.

        if func == 'ALL_MEMBERS_HAVE_SAME_VALUE':
            # args: [ {group_by...}, "AssignedMonth" ]
            # We assume context is already grouped.
            target = node.get('arguments', [])[1]
            if target == "AssignedMonth":
                pod_vars = [self.pod_vars[p] for p in context['pods']]
                if not pod_vars: return self.model.NewConstant(1)

                # Enforce: v[0] == v[1] == v[2]...
                # Returns BoolVar representing compliance
                # Optimization: Direct Add() is faster than reified if strictly enforced
                # But to keep recursive structure, we return a BoolVar
                all_same = self.model.NewBoolVar('all_same')

                # Logic: (v0==v1) AND (v1==v2) ...
                # Easier: All equal to v0
                first = pod_vars[0]
                bools = []
                for other in pod_vars[1:]:
                    b = self.model.NewBoolVar('')
                    self.model.Add(first == other).OnlyEnforceIf(b)
                    self.model.Add(first != other).OnlyEnforceIf(b.Not())
                    bools.append(b)

                if not bools: return self.model.NewConstant(1)
                self.model.AddBoolAnd(bools).OnlyEnforceIf(all_same)
                self.model.AddBoolOr([b.Not() for b in bools]).OnlyEnforceIf(all_same.Not())
                return all_same

        if func == 'GET_POD_COUNT_FOR_MONTH':
            # args: [MonthIndex, OptionalFilter]
            args = node.get('arguments', [])
            m_idx = args[0] # Should be int
            filter_node = args[1] if len(args) > 1 else None

            # Iterate ALL pods (Global calculation, regardless of context scope usually)
            # But technically it returns an INT (count).
            count_var = self.model.NewIntVar(0, len(self.pod_vars), 'count')

            relevant_indicators = []
            for p, v in self.pod_vars.items():
                # Get indicator for this month
                # Check Filter
                is_match = True
                if filter_node:
                    # Eval filter for SINGLE pod context
                    # This is expensive. We need a way to eval boolean logic per pod.
                    # Hack: _eval returns list of values for current context.
                    # We need to switch context to single pod.
                    sub_ctx = {'pods': [p], 'vars': {}}
                    res = self._eval(filter_node, sub_ctx)
                    # res should be a BoolVar or Const
                    if hasattr(res, 'Index'):
                        # It's a BoolVar. We need to AND it with month indicator
                        pass # Complex
                    else:
                        # Python bool
                        if not res: is_match = False

                if is_match:
                    ind = self.pod_month_indicators.get((p, m_idx))
                    if ind: relevant_indicators.append(ind)

            self.model.Add(count_var == sum(relevant_indicators))
            return count_var

        if func == 'SUM':
            # Sum of a list of variables or values
            args = node.get('arguments', []) # usually column name
            # _eval(column) returns list of values
            # If we sum(AssignedMonth), we sum IntVars
            vals = self._eval(args[0], context) # List
            if vals and hasattr(vals[0], 'Index'):
                return sum(vals) # CP-SAT sum
            return sum(vals)

        return None

    def _eval_operator(self, node, context):
        op = node['operator']
        operands = node.get('operands', [])

        # Lazy Eval for special operators? No, eager for now.
        left = self._eval(operands[0], context)
        right = self._eval(operands[1], context) if len(operands) > 1 else None

        # Helper for Lists (Context has multiple pods)
        # If LHS is list [v1, v2] and RHS is scalar X.
        # "==" -> Are ALL v == X?
        # For now, let's assume operators handle lists by broadcasting or aggregating logic.

        if op == '==':
            # Case 1: Variable List vs Scalar
            if isinstance(left, list) and not isinstance(right, list):
                # E.g. [AssignedMonth_1, AssignedMonth_2] == 12
                # Result is BoolVar (True if ALL match)
                bools = []
                for item in left:
                    if hasattr(item, 'Index'):
                        b = self.model.NewBoolVar('')
                        self.model.Add(item == right).OnlyEnforceIf(b)
                        self.model.Add(item != right).OnlyEnforceIf(b.Not())
                        bools.append(b)
                    else:
                        return self.model.NewConstant(1 if item == right else 0)

                res = self.model.NewBoolVar('')
                self.model.AddBoolAnd(bools).OnlyEnforceIf(res)
                self.model.AddBoolOr([b.Not() for b in bools]).OnlyEnforceIf(res.Not())
                return res

            # Case 2: Scalar vs Scalar
            if hasattr(left, 'Index') or hasattr(right, 'Index'):
                b = self.model.NewBoolVar('')
                self.model.Add(left == right).OnlyEnforceIf(b)
                self.model.Add(left != right).OnlyEnforceIf(b.Not())
                return b
            else:
                return self.model.NewConstant(1 if left == right else 0)

        if op == 'IMPLIES':
            # Result is BoolVar: Left -> Right
            # Left and Right must be BoolVars (or castable)
            res = self.model.NewBoolVar('')
            self.model.AddImplication(left, right).OnlyEnforceIf(res)
            # Reverse: if !res, then Left=1 and Right=0
            self.model.AddBoolAnd([left, right.Not()]).OnlyEnforceIf(res.Not())
            return res

        if op == 'AND':
            # Operands is list of bools
            res = self.model.NewBoolVar('')
            # Need to handle >2 operands if `operands` list is passed
            # Re-eval all operands if > 2
            # Here we assumed binary `left, right` above.
            # Fix: `AND` takes list.
            ops = [self._eval(x, context) for x in operands]
            self.model.AddBoolAnd(ops).OnlyEnforceIf(res)
            self.model.AddBoolOr([o.Not() for o in ops]).OnlyEnforceIf(res.Not())
            return res

        if op == '<=':
            if hasattr(left, 'Index') or hasattr(right, 'Index'):
                b = self.model.NewBoolVar('')
                self.model.Add(left <= right).OnlyEnforceIf(b)
                self.model.Add(left > right).OnlyEnforceIf(b.Not())
                return b
            return self.model.NewConstant(1 if left <= right else 0)

        if op == '>=':
            if hasattr(left, 'Index') or hasattr(right, 'Index'):
                b = self.model.NewBoolVar('')
                self.model.Add(left >= right).OnlyEnforceIf(b)
                self.model.Add(left < right).OnlyEnforceIf(b.Not())
                return b
            return self.model.NewConstant(1 if left >= right else 0)

        return None

    def _extract_schedule(self, solver):
        data = []
        for p, v in self.pod_vars.items():
            data.append({'PODNAME': p, 'AssignedMonth': solver.Value(v)})
        return pd.DataFrame(data)

if __name__ == "__main__":
    pass
