import pandas as pd
from ortools.sat.python import cp_model
from collections import defaultdict
import os
import sys
import traceback
import numpy as np
import json

# --- CONFIGURATION ---
TEMP_KNOWLEDGE_FILE = "_temp_knowledge.json"
POD_DATA_FILE = "adbs_data3.csv"
PLAN_OUTPUT_FILE = "_temp_plan.csv"
PREVIOUS_PLAN_FILE = "_previous_plan.csv"

class SchedulingSolver:
    def __init__(self, pod_data, plan_json):
        self.pod_data = pod_data
        self.pod_data_df = pd.DataFrame(pod_data)

        # --- Data Cleaning ---
        # Added 'offline' to the list of boolean conversions
        bool_cols = ['if_federated', 'Bypass_SSO', 'CRE_FLAG', 'SESSION_COOKIE_TIMEOUT', 'IACS', 'APEX', 'Soak', 'offline']
        for col in bool_cols:
            if col in self.pod_data_df.columns:
                self.pod_data_df[col] = self.pod_data_df[col].apply(lambda x: str(x).lower() in ['true', '1', 'yes', 't'])

        if 'FA_USER_COUNT' in self.pod_data_df.columns:
            self.pod_data_df['FA_USER_COUNT'] = pd.to_numeric(
                self.pod_data_df['FA_USER_COUNT'], errors='coerce'
            ).fillna(0).astype(int)

        defaults = {
            'DB_SIZE': '<1.5TB', 'CUSTOMER_TYPE': 'EXTERNAL', 'TypeF': 'single_cohort',
            'Exadata Name': 'UNKNOWN', 'PATCHING_CADENCE': 'UNKNOWN', 'WAVE': 'UNKNOWN',
            'FAMILY_NAME': 'UNKNOWN', 'REGION': 'UNKNOWN', 'Patching_Slots': 'UNKNOWN',
            'offline': False # Default offline status to False
        }
        for col, default in defaults.items():
            if col not in self.pod_data_df.columns:
                if col == 'Exadata Name' and 'Exadata_Name' in self.pod_data_df.columns:
                    self.pod_data_df['Exadata Name'] = self.pod_data_df['Exadata_Name']
                elif col == 'Exadata Name':
                     self.pod_data_df['Exadata Name'] = 'UNKNOWN'
                else:
                    self.pod_data_df[col] = default

        if 'IDCS_MONTH' in self.pod_data_df.columns:
            self.pod_data_df['IDCS_MONTH'] = pd.to_numeric(self.pod_data_df['IDCS_MONTH'], errors='coerce').fillna(0).astype(int)
        else:
            self.pod_data_df['IDCS_MONTH'] = 0

        self.pod_data = self.pod_data_df.to_dict('records')
        self.global_params = plan_json.get('global_parameters', {})
        self.constraints = plan_json.get('constraints', [])
        self.model = cp_model.CpModel()

        self.pod_vars = {}
        self.pod_month_indicators = {}
        self.pod_map = {pod.get('PODNAME'): pod for pod in self.pod_data if pod.get('PODNAME')}

        self.families = defaultdict(list)
        self.exadata_groups = defaultdict(list)
        self.region_exadata_counts = defaultdict(set)

        # Grouping and Region Analysis
        for pod in self.pod_data:
            pod_name = pod.get('PODNAME')
            if pod_name:
                if pod.get('FAMILY_NAME'): self.families[pod['FAMILY_NAME']].append(pod_name)

                ex_name = pod.get('Exadata Name', 'UNKNOWN')
                region = pod.get('REGION', 'UNKNOWN')

                if ex_name != 'UNKNOWN':
                    self.exadata_groups[ex_name].append(pod_name)
                    self.region_exadata_counts[region].add(ex_name)

        # --- SMALL FAMILY LOGIC ---
        # Identify families with <= Threshold pods (default 2)
        self.small_family_threshold = self.global_params.get('small_family_threshold', 2)
        self.small_family_weight = self.global_params.get('small_family_weight', 100)
        self.small_family_pods = set()

        print(f"\\n--- Analysis: Small Families (<= {self.small_family_threshold} pods) ---")
        small_fam_count = 0
        for fam_name, pods in self.families.items():
            if len(pods) <= self.small_family_threshold:
                for p in pods:
                    self.small_family_pods.add(p)
                small_fam_count += 1
        print(f"   Identified {small_fam_count} small families (Total {len(self.small_family_pods)} pods).")
        print(f"   Priority Weight: {self.small_family_weight} (vs 10 for normal pods)")

        # Calculate Region Weights (Bucketed Priority)
        self.region_weights = {}
        print("\\n--- Region Priority Analysis ---")
        for region, exas in self.region_exadata_counts.items():
            count = len(exas)
            # Bucketing Logic
            if count > 100: weight = 1       # Huge regions (Low priority)
            elif count > 20: weight = 50      # Medium regions
            else: weight = 500     # Small regions (High priority)
            self.region_weights[region] = weight
            print(f"   Region: {region:<10} | Exadatas: {count:<4} | Priority Weight: {weight}")
        print("--------------------------------\\n")

        self.previous_assignments = {}
        try:
            if os.path.exists(POD_DATA_FILE):
                input_df = pd.read_csv(POD_DATA_FILE)
                if 'AssignedMonth' in input_df.columns:
                     self.previous_assignments = pd.Series(
                        input_df.AssignedMonth.values, index=input_df.PODNAME
                    ).to_dict()
        except: pass

    def _get_horizon(self):
        total_months = self.global_params.get("total_duration_months", 48)
        start_month = self.global_params.get("start_month_index", 12)
        return start_month, max(start_month, total_months)

    def solve(self):
        print("Engine: Initializing variables...")
        self._create_variables_and_indicators()

        print("Engine: Translating constraints...")
        self._build_model_from_cdl()

        print("Engine: Setting Objective (High-Stake Makespan + Small Family + Offline Priority)...")
        self._set_objective()

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True

        # --- SPEED OPTIMIZATION SETTINGS ---
        solver.parameters.relative_gap_limit = 0.0001
        solver.parameters.random_seed = 42
        solver.parameters.max_time_in_seconds = 1200
        solver.parameters.num_search_workers = 16
        solver.parameters.max_memory_in_mb = 36000
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
        solver.parameters.symmetry_level = 2
        solver.parameters.presolve_probing_deterministic_time_limit = 30

        # Hinting
        if self.previous_assignments:
            print("Engine: Applying hints...")
            count = 0
            start, end = self._get_horizon()
            for pod_name, val in self.previous_assignments.items():
                if pod_name in self.pod_vars:
                    try:
                        v_int = int(val)
                        if start <= v_int <= end:
                            self.model.AddHint(self.pod_vars[pod_name], v_int)
                            count += 1
                    except: pass
            print(f"Engine: Hints applied for {count} pods.")

        print("\\n--- Starting CP-SAT Solver ---")
        status = solver.Solve(self.model)
        print("--- Solver Finished ---\\n")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"Engine: Solution found! (Status: {solver.StatusName(status)})")
            return self._extract_schedule(solver)
        elif status == cp_model.INFEASIBLE:
            print("Engine: Plan is IMPOSSIBLE.")
            return None
        else:
            print(f"Engine: Timed out. (Status: {solver.StatusName(status)})")
            if status == cp_model.UNKNOWN and solver.ResponseStats().solution_fingerprint:
                 print("Engine: Recovering partial solution...")
                 return self._extract_schedule(solver)
            return None

    def _create_variables_and_indicators(self):
        start_month, max_month = self._get_horizon()
        for pod in self.pod_data:
            pod_name = pod.get('PODNAME')
            if pod_name:
                pod_var = self.model.NewIntVar(start_month, max_month, f"month_{pod_name}")
                self.pod_vars[pod_name] = pod_var

                for m in range(start_month, max_month + 1):
                    indicator = self.model.NewBoolVar(f'i_{pod_name}_m{m}')
                    self.model.Add(pod_var == m).OnlyEnforceIf(indicator)
                    self.model.Add(pod_var != m).OnlyEnforceIf(indicator.Not())
                    self.pod_month_indicators[(pod_name, m)] = indicator
        print(f"Engine: Created {len(self.pod_vars)} variables.")

    def _set_objective(self):
        start_month, max_month = self._get_horizon()

        # 1. Exadata Decommissioning (Weighted by Region Size)
        weighted_exa_end_vars = []
        for exadata_name, pod_list in self.exadata_groups.items():
            if not pod_list: continue

            relevant_vars = [self.pod_vars[p] for p in pod_list if p in self.pod_vars]
            if relevant_vars:
                # Determine Exadata End Month
                exa_end_var = self.model.NewIntVar(start_month, max_month, f'end_exa_{exadata_name}')
                self.model.AddMaxEquality(exa_end_var, relevant_vars)

                # Determine Region Weight
                first_pod = pod_list[0]
                region = self.pod_map[first_pod].get('REGION', 'UNKNOWN')
                weight = self.region_weights.get(region, 1)

                # Apply Weight: (EndMonth * RegionPriority * 100)
                weighted_exa_end_vars.append(exa_end_var * weight * 100)

        # 2. Makespan (Global finish time)
        makespan = self.model.NewIntVar(start_month, max_month, 'makespan')
        self.model.AddMaxEquality(makespan, list(self.pod_vars.values()))

        # 3. Sum of Completion Times (General Front-Loading)
        sum_of_months = sum(self.pod_vars.values())

        # 4. Small Family Prioritization
        small_family_terms = []
        for pod_name in self.small_family_pods:
            if pod_name in self.pod_vars:
                small_family_terms.append(self.pod_vars[pod_name] * self.small_family_weight)

        # 5. Offline Pod Prioritization (NEW)
        # Minimize assignment month for pods where 'offline' == True
        offline_weight = self.global_params.get('offline_weight', 100)
        offline_terms = []
        offline_count = 0
        for pod_name, pod_var in self.pod_vars.items():
            # Check if this pod is marked offline in the data
            if self.pod_map[pod_name].get('offline', False) == True:
                offline_terms.append(pod_var * offline_weight)
                offline_count += 1

        print(f"Engine: Objective Config -> Found {offline_count} Offline pods. Weighted by {offline_weight}")

        # COMPOSITE OBJECTIVE:
        # 1. Decommission Hardware (Highest Priority)
        # 2. Compress Schedule (Makespan)
        # 3. Small Family Prioritization (Medium-High Priority)
        # 4. Offline Pod Prioritization (Medium-High Priority)
        # 5. General Front-loading

        self.model.Minimize(
            sum(weighted_exa_end_vars) +
            (makespan * 500) +
            sum(small_family_terms) +
            sum(offline_terms) +
            (sum_of_months * 10)
        )

        print(f"Engine: Objective set. Makespan Weight: 500, Small Fam Weight: {self.small_family_weight}, Offline Weight: {offline_weight}.")

    def _build_model_from_cdl(self):
        for c in self.constraints:
            try: self._apply_constraint(c)
            except Exception as e: print(f"Warning: Error in constraint '{c.get('rule_type')}': {e}")

    def _apply_constraint(self, constraint):
        params = constraint.get('params', {})
        expr = params.get('expression', {})
        rule_type = constraint['rule_type']

        if rule_type == 'group_concurrency_limit': self._apply_group_concurrency_constraint(params)
        elif rule_type == 'mixed_cohort_sequencing': self._apply_mixed_cohort_constraint(params)
        elif rule_type == 'region_slot_wave_concurrency': self._apply_concurrency_constraint(params)
        elif "ALL_MEMBERS_HAVE_SAME_VALUE" in json.dumps(expr): self._apply_grouping_constraint(expr)
        elif "GET_POD_COUNT_FOR_MONTH" in json.dumps(expr) or "for_each_month" in json.dumps(expr): self._apply_global_constraint(expr)
        else:
             if not self.pod_data: return
             for pod in self.pod_data: self._apply_per_pod_constraint(expr, pod)

    def _apply_group_concurrency_constraint(self, params):
        limit = params.get("limit", 100)
        group_cols = params.get("group_columns", [])
        if not group_cols: return

        df_clean = self.pod_data_df.copy()
        for col in group_cols:
            if col in df_clean.columns: df_clean[col] = df_clean[col].fillna("UNKNOWN").astype(str)
            else: return

        grouped_data = df_clean.groupby(group_cols)['PODNAME'].apply(list).to_dict()
        start_month, max_month = self._get_horizon()
        constraint_count = 0

        for group_keys, pod_names in grouped_data.items():
            if len(pod_names) <= limit: continue

            for month in range(start_month, max_month + 1):
                indicators = []
                for pod_name in pod_names:
                    indicator = self.pod_month_indicators.get((pod_name, month))
                    if indicator is not None: indicators.append(indicator)

                if indicators:
                    self.model.Add(sum(indicators) <= limit)
                    constraint_count += 1
        print(f"        - Applied concurrency limit ({limit}). Constraints created: {constraint_count}.")

    def _apply_concurrency_constraint(self, params):
        limit1 = params.get("limit_before_changeover", 250)
        limit2 = params.get("limit_after_changeover", 500)
        changeover_month = params.get("changeover_month", 17)
        start_month, max_month = self._get_horizon()
        required_cols = ['REGION', 'Patching_Slots', 'PATCHING_CADENCE']

        if not all(col in self.pod_data_df.columns for col in required_cols): return

        df_filled = self.pod_data_df.copy()
        for col in required_cols: df_filled[col] = df_filled[col].fillna("UNKNOWN").astype(str)

        grouped = df_filled.groupby(required_cols)
        constraint_count = 0

        for group_key, group_df in grouped:
            pod_names = group_df['PODNAME'].tolist()
            if len(pod_names) <= min(limit1, limit2): continue

            for month in range(start_month, max_month + 1):
                current_limit = limit1 if month < changeover_month else limit2
                indicators = []
                for pod_name in pod_names:
                    indicator = self.pod_month_indicators.get((pod_name, month))
                    if indicator is not None: indicators.append(indicator)

                if indicators:
                    self.model.Add(sum(indicators) <= current_limit)
                    constraint_count += 1
        print(f"        - Applied region/slot concurrency (Generated {constraint_count} sub-constraints).")

    def _apply_mixed_cohort_constraint(self, params):
        start_month_index, max_month = self._get_horizon()
        possible_starts = [m for m in range(start_month_index - 4, max_month + 1) if m % 3 == 2]

        for family_name, pod_names in self.families.items():
            if not pod_names: continue
            first_pod_data = self.pod_map.get(pod_names[0], {})
            if first_pod_data.get("TypeF") != 'mixed_cohort': continue

            soak_c_pods = []
            cohort_a_pods = []
            cohort_b_pods = []
            non_soak_c_pods = []

            for pod_name in pod_names:
                pod_data = self.pod_map.get(pod_name, {})
                cohort = pod_data.get("FACP_COHORT")
                is_soak = pod_data.get("Soak", False)
                if cohort == 'Cohort C' and is_soak: soak_c_pods.append(pod_name)
                elif cohort == 'Cohort A': cohort_a_pods.append(pod_name)
                elif cohort == 'Cohort B': cohort_b_pods.append(pod_name)
                elif cohort == 'Cohort C' and not is_soak: non_soak_c_pods.append(pod_name)

            valid_family_starts = []
            for s in possible_starts:
                is_valid = True
                if soak_c_pods and s < start_month_index: is_valid = False
                if cohort_a_pods and (s + 1) < start_month_index: is_valid = False
                if cohort_b_pods and (s + 2) < start_month_index: is_valid = False
                if non_soak_c_pods and (s + 3) < start_month_index: is_valid = False
                if soak_c_pods and s > max_month: is_valid = False
                if cohort_a_pods and (s + 1) > max_month: is_valid = False
                if cohort_b_pods and (s + 2) > max_month: is_valid = False
                if non_soak_c_pods and (s + 3) > max_month: is_valid = False
                if is_valid: valid_family_starts.append(s)

            if not valid_family_starts: continue

            f_start_var = self.model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(valid_family_starts),
                f'start_win_{family_name}'
            )

            for p in soak_c_pods:
                if p in self.pod_vars: self.model.Add(self.pod_vars[p] == f_start_var)
            for p in cohort_a_pods:
                if p in self.pod_vars: self.model.Add(self.pod_vars[p] == f_start_var + 1)
            for p in cohort_b_pods:
                if p in self.pod_vars: self.model.Add(self.pod_vars[p] == f_start_var + 2)
            for p in non_soak_c_pods:
                if p in self.pod_vars: self.model.Add(self.pod_vars[p] == f_start_var + 3)

    def _apply_per_pod_constraint(self, expr, pod):
        if expr.get("operator") == "IMPLIES":
            condition_var = self._interpret_expression_as_bool_var(expr["operands"][0], pod)
            if condition_var is not None:
                self._interpret_and_enforce_restriction(expr["operands"][1], pod, condition_var)
        else:
            bool_var = self._interpret_expression_as_bool_var(expr, pod)
            if bool_var is not None:
                self.model.Add(bool_var == 1)

    def _interpret_and_enforce_restriction(self, expr, pod_context, condition_var):
        operator = expr.get("operator")
        operands = expr.get("operands", [])

        if operator == ">=":
            left_var = self._get_value(operands[0], pod_context)
            right_val = self._get_value(operands[1], pod_context)
            if isinstance(left_var, cp_model.IntVar) and isinstance(right_val, int):
                self.model.Add(left_var >= right_val).OnlyEnforceIf(condition_var)
        elif operator == "NOT IN" and isinstance(operands[1], list):
            left_var = self._get_value(operands[0], pod_context)
            right_list = operands[1]
            if isinstance(left_var, cp_model.IntVar):
                for val in right_list:
                    try:
                        self.model.Add(left_var != int(val)).OnlyEnforceIf(condition_var)
                    except (ValueError, TypeError): continue
            else:
                if left_var in right_list:
                    self.model.Add(condition_var == 0)
        elif operator == "==":
            left_val = self._get_value(operands[0], pod_context)
            right_val = self._get_value(operands[1], pod_context)
            if left_val != right_val:
                self.model.Add(condition_var == 0)
        elif operator == "AND":
            for op in operands:
                self._interpret_and_enforce_restriction(op, pod_context, condition_var)
        else:
            bool_restriction = self._interpret_expression_as_bool_var(expr, pod_context)
            if bool_restriction is not None:
                self.model.AddImplication(condition_var, bool_restriction)

    def _apply_global_constraint(self, expr):
        if "for_each_month" in expr:
            self._process_iterative_constraint(expr)
            return
        operator = expr.get("operator")
        if operator == "AND":
            for op in expr.get("operands", []):
                if "for_each_month" in op:
                    self._process_iterative_constraint(op)
                else:
                    bool_var = self._interpret_expression_as_bool_var(op, None)
                    if bool_var is not None:
                        self.model.Add(bool_var == 1)
            return
        bool_var = self._interpret_expression_as_bool_var(expr, None)
        if bool_var is not None:
            self.model.Add(bool_var == 1)

    def _process_iterative_constraint(self, expr):
        rule_to_apply = expr["rule"]
        start_month, max_month = self._get_horizon()
        for month in range(start_month, max_month + 1):
            context = {"CURRENT_MONTH": month}
            bool_var = self._interpret_expression_as_bool_var(rule_to_apply, None, loop_context=context)
            if bool_var is not None:
                self.model.Add(bool_var == 1)

    def _apply_grouping_constraint(self, expr):
        func = expr.get("function")
        if func == "ALL_MEMBERS_HAVE_SAME_VALUE":
            for family_name, pod_names in self.families.items():
                if len(pod_names) > 1:
                    first_pod_data = self.pod_map.get(pod_names[0], {})
                    family_type = first_pod_data.get("TypeF")
                    if family_type == 'single_cohort':
                        first_pod_var = self.pod_vars.get(pod_names[0])
                        if first_pod_var is not None:
                            for i in range(1, len(pod_names)):
                                current_pod_var = self.pod_vars.get(pod_names[i])
                                if current_pod_var is not None:
                                    self.model.Add(current_pod_var == first_pod_var)

    def _is_fixed_to(self, var, value):
        if not isinstance(var, cp_model.IntVar): return False
        try:
            proto = var.Proto()
            if not proto.domain: return False
            return len(proto.domain) == 2 and proto.domain[0] == value and proto.domain[1] == value
        except Exception: return False

    def _interpret_expression_as_bool_var(self, expr, pod_context, loop_context=None):
        if not isinstance(expr, dict): return self.model.NewConstant(1)
        operator = expr.get("operator")
        operands = expr.get("operands", [])

        if operator == "AND":
            result_var = self.model.NewBoolVar('')
            ops = [self._interpret_expression_as_bool_var(op, pod_context, loop_context) for op in operands]
            ops = [op for op in ops if op is not None]
            if not ops: return None
            self.model.AddBoolAnd(ops).OnlyEnforceIf(result_var)
            self.model.AddBoolOr([op.Not() for op in ops]).OnlyEnforceIf(result_var.Not())
            return result_var
        elif operator == "IMPLIES":
            result_var = self.model.NewBoolVar('')
            if len(operands) != 2: return self.model.NewConstant(1)
            left = self._interpret_expression_as_bool_var(operands[0], pod_context, loop_context)
            right = self._interpret_expression_as_bool_var(operands[1], pod_context, loop_context)
            if left is None or right is None: return self.model.NewConstant(1)
            self.model.AddImplication(left, right).OnlyEnforceIf(result_var)
            self.model.AddBoolAnd([left, right.Not()]).OnlyEnforceIf(result_var.Not())
            return result_var
        elif operator in ["==", "!=", ">=", "<=", ">", "<"]:
            if len(operands) != 2: return None
            left = self._get_value(operands[0], pod_context, loop_context)
            right = self._get_value(operands[1], pod_context, loop_context)
            if left is None or right is None: return None
            is_left_py = isinstance(left, (bool, str, int, float))
            is_right_py = isinstance(right, (bool, str, int, float))
            if is_left_py and is_right_py:
                py_result = False
                try:
                    if operator == "==": py_result = (left == right)
                    elif operator == "!=": py_result = (left != right)
                    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                        if operator == ">=": py_result = (left >= right)
                        elif operator == "<=": py_result = (left <= right)
                        elif operator == ">": py_result = (left > right)
                        elif operator == "<": py_result = (left < right)
                except TypeError: return self.model.NewConstant(0)
                return self.model.NewConstant(1 if py_result else 0)
            if not isinstance(left, (int, float, cp_model.IntVar)) or not isinstance(right, (int, float, cp_model.IntVar)):
                return self.model.NewConstant(0)
            result_var = self.model.NewBoolVar('')
            if operator == "==":
                self.model.Add(left == right).OnlyEnforceIf(result_var)
                self.model.Add(left != right).OnlyEnforceIf(result_var.Not())
            elif operator == "!=":
                self.model.Add(left != right).OnlyEnforceIf(result_var)
                self.model.Add(left == right).OnlyEnforceIf(result_var.Not())
            elif operator == ">=":
                self.model.Add(left >= right).OnlyEnforceIf(result_var)
                self.model.Add(left < right).OnlyEnforceIf(result_var.Not())
            elif operator == "<=":
                self.model.Add(left <= right).OnlyEnforceIf(result_var)
                self.model.Add(left > right).OnlyEnforceIf(result_var.Not())
            elif operator == ">":
                self.model.Add(left > right).OnlyEnforceIf(result_var)
                self.model.Add(left <= right).OnlyEnforceIf(result_var.Not())
            elif operator == "<":
                self.model.Add(left < right).OnlyEnforceIf(result_var)
                self.model.Add(left >= right).OnlyEnforceIf(result_var.Not())
            return result_var
        elif operator in ["IN", "NOT IN"]:
            result_var = self.model.NewBoolVar('')
            if len(operands) != 2: return self.model.NewConstant(1)
            left_var = self._get_value(operands[0], pod_context, loop_context)
            right_list = self._get_value(operands[1], pod_context, loop_context)
            if left_var is None or not isinstance(right_list, list): return self.model.NewConstant(1)
            if isinstance(left_var, cp_model.IntVar):
                literals = []
                if operator == "IN":
                    for val in right_list:
                        try:
                            int_val = int(val)
                            b = self.model.NewBoolVar('')
                            self.model.Add(left_var == int_val).OnlyEnforceIf(b)
                            self.model.Add(left_var != int_val).OnlyEnforceIf(b.Not())
                            literals.append(b)
                        except (ValueError, TypeError): continue
                    if literals:
                        self.model.AddBoolOr(literals).OnlyEnforceIf(result_var)
                        self.model.AddBoolAnd([lit.Not() for lit in literals]).OnlyEnforceIf(result_var.Not())
                    else: return self.model.NewConstant(0)
                else:
                    for val in right_list:
                        try:
                            int_val = int(val)
                            b = self.model.NewBoolVar('')
                            self.model.Add(left_var != int_val).OnlyEnforceIf(b)
                            self.model.Add(left_var == int_val).OnlyEnforceIf(b.Not())
                            literals.append(b)
                        except (ValueError, TypeError): continue
                    if literals:
                        self.model.AddBoolAnd(literals).OnlyEnforceIf(result_var)
                        self.model.AddBoolOr([lit.Not() for lit in literals]).OnlyEnforceIf(result_var.Not())
                    else: return self.model.NewConstant(1)
            else:
                py_result = (left_var in right_list) if operator == "IN" else (left_var not in right_list)
                return self.model.NewConstant(1 if py_result else 0)
            return result_var
        return None

    def _get_value(self, operand, pod_context, loop_context=None):
        if isinstance(operand, (int, bool, str, list, float)): return operand
        if not isinstance(operand, dict): return None
        if "variable" in operand:
            var_name = operand["variable"]
            if loop_context and var_name in loop_context: return loop_context[var_name]
            if var_name == "AssignedMonth": return self.pod_vars.get(pod_context.get('PODNAME')) if pod_context else None
            if pod_context: return pod_context.get(var_name)
            return None
        if "global" in operand: return self.global_params.get(operand["global"])
        if "function" in operand:
            func_name = operand["function"]
            args = operand.get("arguments", [])
            if func_name == "GET_POD_COUNT_FOR_MONTH":
                return self._calculate_pod_count_for_month_optimized(args, pod_context, loop_context)
        return None

    def _calculate_pod_count_for_month_optimized(self, args, pod_context, loop_context):
        target_month_val = self._get_value(args[0], pod_context, loop_context)
        if not isinstance(target_month_val, int): return None
        target_month = target_month_val
        filter_expr = args[1] if len(args) > 1 else None
        indicators = []
        for pod in self.pod_data:
            pod_name = pod.get('PODNAME')
            if filter_expr:
                filter_match = self._interpret_expression_as_bool_var(filter_expr, pod)
                if filter_match is None: continue
                if self._is_fixed_to(filter_match, 0): continue
                elif self._is_fixed_to(filter_match, 1):
                    month_indicator = self.pod_month_indicators.get((pod_name, target_month))
                    if month_indicator is not None: indicators.append(month_indicator)
                else:
                    month_indicator = self.pod_month_indicators.get((pod_name, target_month))
                    if month_indicator is not None:
                        aux_name = f'aux_{pod_name}_m{target_month}_f{hash(json.dumps(filter_expr))}'
                        combined_indicator = self.model.NewBoolVar(aux_name)
                        self.model.AddBoolAnd([month_indicator, filter_match]).OnlyEnforceIf(combined_indicator)
                        self.model.AddBoolOr([month_indicator.Not(), filter_match.Not()]).OnlyEnforceIf(combined_indicator.Not())
                        indicators.append(combined_indicator)
            else:
                month_indicator = self.pod_month_indicators.get((pod_name, target_month))
                if month_indicator is not None: indicators.append(month_indicator)
        sum_var_name = f'count_m_{target_month}'
        if filter_expr: sum_var_name += f'_f_{hash(json.dumps(filter_expr))}'
        sum_var = self.model.NewIntVar(0, len(self.pod_data), sum_var_name)
        self.model.Add(sum_var == sum(indicators))
        return sum_var

    def _extract_schedule(self, solver):
        schedule = []
        for pod_name, var in self.pod_vars.items():
            assigned_month = solver.Value(var)
            pod_info = self.pod_map.get(pod_name, {})
            ex_name = pod_info.get("Exadata_Name") or pod_info.get("Exadata Name") or "N/A"

            schedule.append({
                "PODNAME": pod_name,
                "AssignedMonth": assigned_month,
                "Exadata_Name": ex_name,
                "DB_SIZE": pod_info.get("DB_SIZE", "N/A"),
                "REGION": pod_info.get("REGION", "N/A"),
                "FAMILY_NAME": pod_info.get("FAMILY_NAME", "N/A"),
                "CUSTOMER_TYPE": pod_info.get("CUSTOMER_TYPE", "N/A"),
                "TypeF": pod_info.get("TypeF", "N/A"),
                "FACP_COHORT": pod_info.get("FACP_COHORT", "N/A")
            })
        return pd.DataFrame(schedule)

def run():
    try:
        print("Solver script started (With Regional Prioritization + Small Family + Offline Optimization).")
        if not os.path.exists(POD_DATA_FILE):
            print(f"Warning: Pod data file not found at {POD_DATA_FILE}.", file=sys.stderr)
            sys.exit(1)

        try: pod_data_df = pd.read_csv(POD_DATA_FILE)
        except Exception as e:
            print(f"Error reading CSV file: {e}", file=sys.stderr)
            sys.exit(1)

        pod_data_list = pod_data_df.to_dict('records')
        print(f"Pod data loaded. Total pods: {len(pod_data_list)}")

        knowledge_file_to_load = TEMP_KNOWLEDGE_FILE
        if not os.path.exists(TEMP_KNOWLEDGE_FILE):
            master_knowledge = "knowledge_adbs5.json"
            if os.path.exists(master_knowledge): knowledge_file_to_load = master_knowledge
            else:
                print(f"ERROR: Knowledge file not found.", file=sys.stderr)
                sys.exit(1)

        print(f"Reading constraints from: {knowledge_file_to_load}")
        with open(knowledge_file_to_load, 'r') as f: current_plan = json.load(f)

        if not pod_data_list:
            final_schedule_df = pd.DataFrame(columns=['PODNAME', 'AssignedMonth'])
        else:
            solver_engine = SchedulingSolver(pod_data_list, current_plan)
            final_schedule_df = solver_engine.solve()

        if final_schedule_df is not None:
            print(f"Writing plan to: {PLAN_OUTPUT_FILE}")
            final_schedule_df.to_csv(PLAN_OUTPUT_FILE, index=False)
            print("Plan written successfully.")
        else:
            print("Solver failed. No output file written.")
            sys.exit(1)

    except Exception:
        print("--- A FATAL ERROR OCCURRED IN THE SOLVER SCRIPT ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run()
