import pandas as pd

class SchemaEngine:
    @staticmethod
    def generate_passport(df):
        """
        Analyzes the 'Physical Reality' of the uploaded data.
        Returns a text summary injected into the LLM Prompt.
        """
        passport = ["### DATA PASSPORT (The Reality of this Dataset)"]
        passport.append(f"TOTAL ROWS: {len(df)}")
        passport.append("COLUMNS & PROFILE:")

        for col in df.columns:
            # 1. Numeric Columns
            if pd.api.types.is_numeric_dtype(df[col]):
                min_v, max_v = df[col].min(), df[col].max()
                passport.append(f"- **{col}** (Numeric): Range [{min_v} to {max_v}]")
            
            # 2. Categorical / Text Columns
            else:
                unique = df[col].dropna().unique().tolist()
                count = len(unique)
                
                if count < 50:
                    # Low Cardinality: List all values (Enables Semantic Mapping)
                    # e.g. Maps "Small" -> "<1.5TB"
                    vals = ", ".join(sorted(map(str, unique)))
                    passport.append(f"- **{col}** (Category): Values [{vals}]")
                else:
                    # High Cardinality: Just identifying type (Prevents Context Overflow)
                    # e.g. Exadata Names, Pod IDs
                    example = unique[0] if unique else "N/A"
                    passport.append(f"- **{col}** (High Cardinality ID): {count} unique values. Ex: '{example}'...")

        return "\n".join(passport)