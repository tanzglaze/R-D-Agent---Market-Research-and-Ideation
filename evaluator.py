"""
Evaluator
Parses the execution output to extract a numeric score.
Expects the generated code to print a line like:
  SCORE: 0.9523
"""

import re
from typing import Optional
from core.executor import ExecutionResult


class Evaluator:
    SCORE_PATTERN = re.compile(r"SCORE:\s*([\d.]+)", re.IGNORECASE)

    def evaluate(self, result: ExecutionResult) -> Optional[float]:
        """
        Extract the SCORE value from execution output.
        Returns None if the run failed or no score was printed.
        """
        if not result.success:
            return None

        match = self.SCORE_PATTERN.search(result.stdout)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

        return None

    def is_improvement(
        self, new_score: Optional[float], best_score: Optional[float]
    ) -> bool:
        """Return True if new_score is better than best_score."""
        if new_score is None:
            return False
        if best_score is None:
            return True
        return new_score > best_score

    def format_feedback(
        self,
        result: ExecutionResult,
        score: Optional[float],
        best_score: Optional[float],
    ) -> str:
        """Build a feedback string for the next Research Agent prompt."""
        lines = []

        if result.timed_out:
            lines.append("The code timed out. Simplify the model or reduce iterations.")
        elif not result.success:
            lines.append(f"The code crashed with error:\n{result.stderr[:500]}")
        elif score is None:
            lines.append(
                "The code ran but did not print 'SCORE: <value>'. "
                "Make sure to print the accuracy score in that exact format."
            )
        else:
            lines.append(f"The code achieved a score of {score:.4f}.")
            if best_score is not None:
                if score > best_score:
                    lines.append(
                        f"This is an improvement over the previous best of {best_score:.4f}!"
                    )
                else:
                    lines.append(
                        f"This did NOT improve over the best score of {best_score:.4f}. "
                        "Try a different approach."
                    )

        return "\n".join(lines)
