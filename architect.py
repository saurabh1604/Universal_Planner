import openai
import json
import pandas as pd
from rapidfuzz import process, fuzz

class GenericArchitect:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

    def generate_cdl(self, user_input, data_passport, df=None):
        """
        Translates NL -> Strict Mathematical CDL -> Validates Values
        """
        
        # 1. THE STRICT SYSTEM PROMPT
        system_prompt = f"""
        You are the CDL Architect. Convert Natural Language into a STRICT Recursive Mathematical Expression Tree.
        
        {data_passport}

        ### GRAMMAR RULES (The 4 Primitive Buckets)
        1. **Filtering** (Where): {{ "operator": "==", "operands": [{{ "variable": "REGION" }}, "PHX"] }}
        2. **Temporal** (When): {{ "operator": ">=", "operands": [{{ "variable": "AssignedMonth" }}, 15] }}
        3. **Aggregation** (Sum/Count): 
           - Use "function": "SUM" inside an operand.
           - Args: ["count" OR "COLUMN_NAME", Filter_Expression_OR_null]
        4. **Scope** (Grouping):
           - If user says "Per Exadata" or "Per Region", use "scope": {{ "type": "FOR_EACH_UNIQUE_COMBINATION", "columns": ["Exadata_Name"] }}

        ### STRICT JSON OUTPUT FORMAT
        {{
            "rule_type": "descriptive_snake_case",
            "description": "Human readable description",
            "params": {{
                "scope": {{ "type": "GLOBAL" }} OR {{ "type": "FOR_EACH_UNIQUE_COMBINATION", "columns": [...] }},
                "expression": {{
                    "operator": "AND/OR/IMPLIES/==/<=/IN",
                    "operands": [ ... recursive nesting ... ]
                }}
            }}
        }}

        ### CRITICAL INSTRUCTIONS
        - Map user terms (e.g., "Tiny") to Schema Values (e.g., "<1.5TB").
        - DO NOT use high-level functions like "Groupby_Pods". Use primitives.
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
        Handles High Cardinality columns (like Exadata Names) that weren't in the Passport.
        """
        expr = cdl_json.get('params', {}).get('expression', {})
        
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
            return None

        return recursive_check(expr)