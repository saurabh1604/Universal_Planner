import openai
import json
import pandas as pd
import os
from rapidfuzz import process, fuzz

class GenericArchitect:
    def __init__(self, api_key=None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
             self.client = None
        else:
             self.client = openai.OpenAI(api_key=key)

    def generate_cdl(self, user_input, data_passport, df=None, chat_history=None, plan_start_date=None):
        """
        Translates NL -> Strict Mathematical CDL -> Validates Values
        """
        if not self.client:
             return None, "OpenAI API Key is missing. Please set HARDCODED_API_KEY in app.py or OPENAI_API_KEY env var."

        # Format chat history
        history_context = ""
        if chat_history:
            relevant_history = chat_history[-6:]
            history_context = "\n### CONVERSATION HISTORY (For Context):\n"
            for msg in relevant_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_context += f"- {role.upper()}: {content}\n"

            history_context += "\nIMPORTANT: Use the HISTORY to understand if the user is clarifying an earlier ambiguous term (e.g. defining 'Large' size). Combine the original intent with the clarification."

        # Plan Start Date Context
        start_date_context = "- **Plan Start Date**: Dec 2025 is Month 12. Jan 2026 is Month 13. Jan 2027 is Month 25. (DEFAULT)"
        if plan_start_date:
            start_date_context = f"- **Plan Start Date (Month 1)**: {plan_start_date}. Calculate all relative Month Indices based on this anchor."

        # 1. THE STRICT SYSTEM PROMPT
        system_prompt = f"""
        You are the CDL Architect. Convert Natural Language into a STRICT Recursive Mathematical Expression Tree.
        
        {data_passport}

        ### VIRTUAL VARIABLES (Known to Solver)
        - **AssignedMonth** (Integer): The target month index for scheduling.
        - **CURRENT_MONTH** (Integer): Loop variable if using iterators.
        {start_date_context}

        ### GRAMMAR RULES (Primitive Buckets)
        1. **Filtering** (Where): {{ "operator": "==", "operands": [{{ "variable": "REGION" }}, "PHX"] }}
        2. **Temporal** (When): {{ "operator": ">=", "operands": [{{ "variable": "AssignedMonth" }}, 15] }}
        3. **Arithmetic** (Math): Support `+`, `-`, `*`, `/`, `%` (Modulo).
           - Example: {{ "operator": "%", "operands": [{{ "variable": "AssignedMonth" }}, 3] }}
        4. **Aggregation** (Sum/Count):
           - Use "function": "SUM" inside an operand.
           - Args: ["count" OR "COLUMN_NAME", Filter_Expression_OR_null]
        5. **Scope** (Grouping):
           - If user says "Per Exadata" or "Per Region", use "scope": {{ "type": "FOR_EACH_UNIQUE_COMBINATION", "columns": ["Exadata_Name"] }}

        ### ADVANCED SOLVER FUNCTIONS (Use these for specific logic)
        1. **Iterators (Loops)**:
           - Use `iterator` to apply a rule multiple times (e.g. "For every month").
           - Params: `{{ "variable": "m", "range": [1, 48] }}` (or `"range": "ALL_MONTHS"`).
           - Inside expression, use `{{ "variable": "m" }}`.

        2. **Cohesion / Grouping**:
           - Function: `ALL_MEMBERS_HAVE_SAME_VALUE`
           - Arguments: ["AssignedMonth"] (or expression like `{{ "operator": "-", "operands": ["AssignedMonth", "Offset"] }}`)
           - Requirement: Must be used with a `scope`.

        3. **Capacity / Ramp-up**:
           - Function: `GET_SUM_FOR_MONTH` or `GET_POD_COUNT_FOR_MONTH`
           - Arguments: [MonthIndex, ColumnName (or "count"), OptionalFilter]

        ### OBJECTIVES (Soft Constraints)
        - If user says "Prioritize", "Front-load", "Minimize delay", "Prefer":
          - Set `params.type` to "OBJECTIVE".
          - `params.action`: "MINIMIZE" (default) or "MAXIMIZE".
          - `params.weight`: Priority (e.g. 100, 500, 1000).
          - `params.expression`: The numeric value to optimize (usually `{{ "variable": "AssignedMonth" }}`).
          - `params.scope`: Define WHICH pods this applies to (Scope + Filter).

        ### STRICT JSON OUTPUT FORMAT
        {{
            "rule_type": "descriptive_snake_case",
            "description": "Human readable description",
            "params": {{
                "type": "CONSTRAINT" (default) OR "OBJECTIVE",
                "action": "MINIMIZE" OR "MAXIMIZE" (Only for Objective),
                "weight": 100 (Only for Objective),
                "iterator": {{ "variable": "m", "range": "ALL_MONTHS" }} (OPTIONAL),
                "scope": {{ "type": "GLOBAL" }} OR {{ "type": "FOR_EACH_UNIQUE_COMBINATION", "columns": [...] }},
                "expression": {{
                    "operator": "AND/OR/IMPLIES/==/<=/IN/%/+",
                    "operands": [ ... recursive nesting ... ]
                }}
            }}
        }}

        ### FEW-SHOT EXAMPLES (LEARN THE STRUCTURE)

        Example 1: "Prioritize small families (Size < 2TB) to start early"
        {{
          "rule_type": "prioritize_small_families",
          "params": {{
            "type": "OBJECTIVE",
            "action": "MINIMIZE",
            "weight": 100,
            "scope": {{
                "type": "GLOBAL",
                "filter": {{ "operator": "<", "operands": [ {{ "variable": "DB_SIZE" }}, 2000 ] }}
            }},
            "expression": {{ "variable": "AssignedMonth" }}
          }}
        }}

        Example 2: "Limit pods to 200 per Exadata per month"
        {{
          "rule_type": "group_concurrency_limit",
          "params": {{
            "iterator": {{ "variable": "m", "range": "ALL_MONTHS" }},
            "scope": {{ "type": "FOR_EACH_UNIQUE_COMBINATION", "columns": ["Exadata Name"] }},
            "expression": {{
                "operator": "<=",
                "operands": [
                    {{ "function": "GET_POD_COUNT_FOR_MONTH", "arguments": [{{ "variable": "m" }}, "count"] }},
                    200
                ]
            }}
          }}
        }}

        Example 3: "Mixed Cohort: If Cohort C (Soak), must start in Month where (Month % 3 == 2)"
        {{
          "rule_type": "mixed_cohort_soak_modulo",
          "params": {{
            "expression": {{
              "operator": "IMPLIES",
              "operands": [
                {{
                  "operator": "AND",
                  "operands": [
                    {{ "operator": "==", "operands": [{{ "variable": "FACP_COHORT" }}, "Cohort C"] }},
                    {{ "operator": "==", "operands": [{{ "variable": "Soak" }}, true] }}
                  ]
                }},
                {{
                   "operator": "==",
                   "operands": [
                      {{ "operator": "%", "operands": [{{ "variable": "AssignedMonth" }}, 3] }},
                      2
                   ]
                }}
              ]
            }}
          }}
        }}

        ### CRITICAL INSTRUCTIONS
        - Use `IMPLIES` for conditional logic.
        - Use `iterator` for "Per Month" constraints.
        - Use `OBJECTIVE` type for priorities/preferences.
        - Calculate Month Indices relative to Plan Start Date.

        {history_context}
        """

        # 2. CALL LLM
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            cdl_json = json.loads(response.choices[0].message.content)
            
            # 3. JIT VALIDATION (The Safety Net)
            if df is not None:
                validation_error = self._validate_values(cdl_json, df)
                if validation_error:
                    return None, validation_error
            
            return cdl_json, None

        except Exception as e:
            return None, str(e)

    def _validate_values(self, cdl_json, df):
        """
        Recursively checks if values in the CDL actually exist in the DataFrame.
        """
        expr = cdl_json.get('params', {}).get('expression', {})
        scope = cdl_json.get('params', {}).get('scope', {})
        
        def recursive_check(node):
            if not isinstance(node, dict): return None
            
            # Check "Variable == Value" patterns
            if 'operator' in node and 'operands' in node:
                op = node['operator']
                args = node['operands']
                
                # Identify Variable and Value
                col_name = None
                target_vals = []
                
                for arg in args:
                    if isinstance(arg, dict) and 'variable' in arg:
                        col_name = arg['variable']
                    elif isinstance(arg, (str, int, float)):
                        target_vals.append(arg)
                    elif isinstance(arg, list): # For IN operator
                        target_vals.extend(arg)
                
                # VALIDATE
                if col_name and target_vals:
                    # Skip validation for Virtual Variables
                    if col_name in ["AssignedMonth", "CURRENT_MONTH", "m", "i"]:
                        return None

                    if col_name not in df.columns:
                        return f"Column '{col_name}' not found in data."
                    
                    # If column is text, check values
                    if df[col_name].dtype == 'object':
                        real_values = set(df[col_name].dropna().unique())
                        for val in target_vals:
                            if val not in real_values:
                                # Fuzzy Match Suggestion
                                best_match = process.extractOne(val, real_values, scorer=fuzz.ratio)
                                msg = f"Value '{val}' not found in '{col_name}'."
                                if best_match and best_match[1] > 80:
                                    msg += f" Did you mean '{best_match[0]}'?"
                                return msg
            
            # Recurse
            for arg in node.get('operands', []):
                if isinstance(arg, dict):
                    err = recursive_check(arg)
                    if err: return err
            # Also recurse into function arguments
            if "arguments" in node:
                for arg in node["arguments"]:
                    if isinstance(arg, dict):
                         err = recursive_check(arg)
                         if err: return err
            return None

        # Check Expression
        err = recursive_check(expr)
        if err: return err

        # Check Scope Filter
        if 'filter' in scope:
            err = recursive_check(scope['filter'])
            if err: return err

        return None
