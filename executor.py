"""
Code Executor
Runs LLM-generated Python code in a subprocess with a timeout.
Captures stdout, stderr, and exit code.
"""

import subprocess
import tempfile
import os
import textwrap
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[STDERR]\n{self.stderr.strip()}")
        if self.timed_out:
            parts.append("[TIMEOUT]")
        return "\n".join(parts)


class Executor:
    def __init__(self, timeout: int = 60):
        """
        Args:
            timeout: Max seconds to let generated code run.
        """
        self.timeout = timeout

    def run(self, code: str) -> ExecutionResult:
        """
        Write code to a temp file and execute it.
        Returns stdout, stderr, exit code.
        """
        # Clean up any markdown fences the LLM might have included
        code = self._clean_code(code)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="rdagent_"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )
        finally:
            os.unlink(tmp_path)

    def _clean_code(self, code: str) -> str:
        """Strip markdown code fences if the LLM included them."""
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        return textwrap.dedent(code)
