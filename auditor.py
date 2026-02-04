import openai
import json
import os

class Auditor:
    def __init__(self, api_key=None):
        # Use provided key, or fall back to environment variable, or raise error if critical
        # The user wants it to work in their system, so we check env var too.
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
             # If no key is found, we can't initialize client correctly, but we won't crash yet.
             # We'll fail at runtime.
             self.client = None
        else:
             self.client = openai.OpenAI(api_key=key)

    def audit_request(self, user_input, data_passport, existing_constraints, chat_history=None):
        """
        Analyzes the request for Hallucinations, Logic Conflicts, and Ambiguity.
        Returns: (status, message)
        Status: "PASS", "FAIL", "CLARIFICATION_NEEDED"
        """
        if not self.client:
             return "FAIL", "OpenAI API Key is missing. Please set HARDCODED_API_KEY in app.py or OPENAI_API_KEY env var."

        # Format chat history for context if available
        history_context = ""
        if chat_history:
            # We only take the last few turns to avoid context bloat, but enough for clarification
            relevant_history = chat_history[-6:]
            history_context = "\n### CONVERSATION HISTORY (For Context):\n"
            for msg in relevant_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_context += f"- {role.upper()}: {content}\n"

        system_prompt = f"""
        You are the Auditor Agent. Your job is to validate user requests before they are turned into code.

        ### DATA PASSPORT (The Reality):
        {data_passport}

        ### EXISTING CONSTRAINTS (The Rules):
        {json.dumps(existing_constraints, indent=2)}

        {history_context}

        ### CHECKS TO PERFORM:
        1. **Hallucination Check**: Does the user mention columns or values that do not exist in the Passport?
        2. **Logic Check**: Does the request contradict any EXISTING CONSTRAINTS? (e.g. "Start in Jan" vs "Blackout Jan")
        3. **Ambiguity Check**: Are terms vague? (e.g. "Move the big ones" -> Ask "By big, what DB_SIZE do you mean?")
           *NOTE: If the user is responding to a clarification question (see HISTORY), use their answer to resolve the ambiguity.*

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
