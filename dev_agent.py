"""
Dev Agent
Takes a hypothesis from the Research Agent and writes
executable Python code that implements and evaluates it.
"""

from openai import OpenAI


class DevAgent:
    SYSTEM_PROMPT = """You are a Python machine learning engineer.
Your job is to write a complete, runnable Python script that implements a given hypothesis on a dataset.

Rules:
- The script must be completely self-contained (no user input, no file arguments).
- Load the dataset using the exact method described in the task.
- Implement the hypothesis exactly as described.
- Evaluate using cross-validation (cv=5) and print the mean accuracy in this EXACT format:
    SCORE: 0.9523
- Use only: scikit-learn, pandas, numpy (always available).
- Do NOT include markdown fences (``` or ```python) — output raw Python only.
- Keep the code under 60 lines.
"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def implement(
        self,
        hypothesis: str,
        task_description: str,
        data_loading_code: str,
        last_error: str = "",
    ) -> str:
        """
        Returns a Python code string that implements the hypothesis.
        """
        error_note = ""
        if last_error:
            error_note = f"\nImportant: The previous code had this error — fix it:\n{last_error[:300]}"

        user_message = f"""
Task: {task_description}

Data loading (use this exactly):
{data_loading_code}

Hypothesis to implement: {hypothesis}
{error_note}

Write the complete Python script now.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,  # low temperature for reliable code generation
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()
