"""
Research Agent
Uses an LLM to propose data science hypotheses based on:
  - The task description
  - Past experiment results from the Knowledge Store
  - The last feedback from the Evaluator
"""

from openai import OpenAI


class ResearchAgent:
    SYSTEM_PROMPT = """You are a data science research assistant.
Your job is to propose ONE clear, actionable hypothesis for improving an ML model on a classification task.

Rules:
- Be specific: name the exact algorithm, feature engineering step, or preprocessing technique.
- Be concise: one sentence, no more.
- Build on past experiments — don't repeat what already failed.
- Your hypothesis will be handed to a developer who will implement it in Python.

Examples of good hypotheses:
- "Use a Random Forest with 200 trees and max_depth=10 instead of a default Decision Tree."
- "Add polynomial features of degree 2 before fitting a Logistic Regression."
- "Scale features with StandardScaler and use SVM with RBF kernel and C=10."
"""

    def __init__(self, client: OpenAI, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def propose(
        self,
        task_description: str,
        past_experiments_summary: str,
        last_feedback: str,
        iteration: int,
    ) -> str:
        """
        Returns a single hypothesis string.
        """
        user_message = f"""
Task: {task_description}

Past experiments:
{past_experiments_summary}

Last feedback:
{last_feedback}

Iteration: {iteration}

Propose one new hypothesis to try next.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.9,  # slightly creative for idea generation
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()
