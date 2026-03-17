"""
Knowledge Store
Persists all experiment results so agents can learn from past iterations.
"""

import json
import os
from datetime import datetime
from typing import Optional


class KnowledgeStore:
    def __init__(self, path: str = "logs/knowledge.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.experiments: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def add(
        self,
        iteration: int,
        hypothesis: str,
        code: str,
        output: str,
        score: Optional[float],
        error: Optional[str] = None,
    ):
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "code": code,
            "output": output,
            "score": score,
            "error": error,
        }
        self.experiments.append(entry)
        self._save()
        return entry

    def best(self) -> Optional[dict]:
        """Return the experiment with the highest score."""
        scored = [e for e in self.experiments if e["score"] is not None]
        if not scored:
            return None
        return max(scored, key=lambda e: e["score"])

    def recent(self, n: int = 3) -> list[dict]:
        """Return the N most recent experiments."""
        return self.experiments[-n:]

    def summary(self) -> str:
        """Human-readable summary of past experiments for the LLM."""
        if not self.experiments:
            return "No experiments yet."

        lines = []
        for e in self.experiments[-5:]:  # last 5 only to keep prompts short
            score_str = f"{e['score']:.4f}" if e["score"] is not None else "FAILED"
            lines.append(
                f"  - Iteration {e['iteration']}: score={score_str} | "
                f"hypothesis=\"{e['hypothesis'][:80]}...\""
            )

        best = self.best()
        if best:
            lines.append(
                f"\n  Best so far: score={best['score']:.4f} at "
                f"iteration {best['iteration']}"
            )

        return "\n".join(lines)
