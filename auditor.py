import openai
import json

class Auditor:
    def __init__(self, api_key=None, mock_mode=False):
        self.client = None
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        self.mock_mode = mock_mode

    def audit_request(self, user_input, data_passport, existing_constraints):
        """
        Analyzes the request for Hallucinations, Logic Conflicts, and Ambiguity.
        Returns: (status, message)
        Status: "PASS", "FAIL", "CLARIFICATION_NEEDED"
        """
        if self.mock_mode or not self.client:
            return self._mock_audit(user_input, existing_constraints)

        system_prompt = f"""
        You are the Auditor Agent. Your job is to validate user requests before they are turned into code.

        ### DATA PASSPORT (The Reality):
        {data_passport}

        ### EXISTING CONSTRAINTS (The Rules):
        {json.dumps(existing_constraints, indent=2)}

        ### CHECKS TO PERFORM:
        1. **Hallucination Check**: Does the user mention columns or values that do not exist in the Passport?
        2. **Logic Check**: Does the request contradict any EXISTING CONSTRAINTS? (e.g. "Start in Jan" vs "Blackout Jan")
        3. **Ambiguity Check**: Are terms vague? (e.g. "Move the big ones" -> Ask "By big, what DB_SIZE do you mean?")

        ### OUTPUT FORMAT (JSON ONLY):
        {{
            "status": "PASS" | "FAIL" | "CLARIFICATION_NEEDED",
            "message": "Reason for failure, or the clarification question, or 'OK'."
        }}
        """

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
            result = json.loads(response.choices[0].message.content)
            return result["status"], result["message"]
        except Exception as e:
            return "FAIL", f"Auditor Error: {str(e)}"

    def _mock_audit(self, user_input, existing_constraints):
        """
        Simulated Logic for testing without API Key.
        """
        u_in = user_input.lower()

        # 1. Ambiguity Trigger
        if "big" in u_in or "small" in u_in:
            return "CLARIFICATION_NEEDED", "By 'big' or 'small', what specific DB_SIZE range do you mean?"

        # 2. Logic Trigger (Contradiction)
        # Check if we are asking for "Jan" but "Jan" is already restricted in existing constraints (simulated)
        for c in existing_constraints:
            desc = c.get("description", "").lower()
            if "blackout" in desc and "jan" in desc and "jan" in u_in:
                 return "FAIL", "Request conflicts with existing rule: 'Blackout Jan'."

        # 3. Hallucination Trigger
        if "unicorn" in u_in:
             return "FAIL", "Column 'Unicorn' does not exist in the Data Passport."

        return "PASS", "Request is valid."
