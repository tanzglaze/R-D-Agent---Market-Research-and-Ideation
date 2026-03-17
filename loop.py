"""
R&D Loop Orchestrator
Ties together: Research Agent → Dev Agent → Executor → Evaluator → repeat.

Usage:
    python loop.py --task iris --iterations 5
    python loop.py --task wine --iterations 3
"""

import argparse
import os
import sys
import json
from datetime import datetime

from openai import OpenAI

from agents.research_agent import ResearchAgent
from agents.dev_agent import DevAgent
from core.executor import Executor
from core.evaluator import Evaluator
from core.knowledge_store import KnowledgeStore


# ── ANSI colour helpers (work on most terminals) ──────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(text):  print(f"\n{BOLD}{CYAN}{'='*60}{RESET}\n{BOLD}{CYAN}{text}{RESET}\n{BOLD}{CYAN}{'='*60}{RESET}")
def section(text): print(f"\n{BOLD}{YELLOW}── {text} ──{RESET}")
def success(text): print(f"{GREEN}✓ {text}{RESET}")
def error(text):   print(f"{RED}✗ {text}{RESET}")
def info(text):    print(f"  {text}")


def load_task(task_name: str):
    """Load task description and data loading code by name."""
    if task_name == "iris":
        from tasks.iris_task import TASK_DESCRIPTION, DATA_LOADING_CODE
    elif task_name == "wine":
        from tasks.wine_task import TASK_DESCRIPTION, DATA_LOADING_CODE
    else:
        print(f"Unknown task '{task_name}'. Available: iris, wine")
        sys.exit(1)
    return TASK_DESCRIPTION.strip(), DATA_LOADING_CODE.strip()


def run_loop(task_name: str, iterations: int, api_key: str):
    header(f"🤖 RD-Agent — Task: {task_name.upper()}   Iterations: {iterations}")

    # ── Initialise components ─────────────────────────────────────────────────
    client = OpenAI(api_key=api_key)
    research_agent = ResearchAgent(client)
    dev_agent      = DevAgent(client)
    executor       = Executor(timeout=60)
    evaluator      = Evaluator()
    store          = KnowledgeStore(path=f"logs/{task_name}_knowledge.json")

    task_description, data_loading_code = load_task(task_name)

    last_feedback = "No experiments yet. Start with a simple baseline."
    best_score: float | None = None

    # ── Main iteration loop ───────────────────────────────────────────────────
    for i in range(1, iterations + 1):
        header(f"Iteration {i} / {iterations}")

        # 1. RESEARCH: propose hypothesis
        section("Research Agent — proposing hypothesis")
        hypothesis = research_agent.propose(
            task_description=task_description,
            past_experiments_summary=store.summary(),
            last_feedback=last_feedback,
            iteration=i,
        )
        info(f"Hypothesis: {hypothesis}")

        # 2. DEVELOP: write code
        section("Dev Agent — writing code")
        last_error = ""
        code = dev_agent.implement(
            hypothesis=hypothesis,
            task_description=task_description,
            data_loading_code=data_loading_code,
        )
        info(f"Code written ({len(code.splitlines())} lines)")

        # 3. EXECUTE (with one retry on error)
        section("Executor — running code")
        result = executor.run(code)

        if not result.success:
            info("First run failed — retrying with error context...")
            last_error = result.stderr or "Unknown error"
            code = dev_agent.implement(
                hypothesis=hypothesis,
                task_description=task_description,
                data_loading_code=data_loading_code,
                last_error=last_error,
            )
            result = executor.run(code)

        if result.success:
            success("Code ran successfully")
            info(result.stdout.strip()[:300])
        else:
            error("Code failed after retry")
            info(result.combined_output[:300])

        # 4. EVALUATE: extract score
        section("Evaluator — scoring")
        score = evaluator.evaluate(result)
        if score is not None:
            improvement = evaluator.is_improvement(score, best_score)
            if improvement:
                best_score = score
                success(f"New best score: {score:.4f} 🎉")
            else:
                info(f"Score: {score:.4f}  (best: {best_score:.4f})")
        else:
            error("Could not extract score from output")

        # 5. Build feedback for next iteration
        last_feedback = evaluator.format_feedback(result, score, best_score)

        # 6. Save to knowledge store
        store.add(
            iteration=i,
            hypothesis=hypothesis,
            code=code,
            output=result.combined_output,
            score=score,
            error=result.stderr if not result.success else None,
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    header("Final Summary")
    best = store.best()
    if best:
        success(f"Best score achieved: {best['score']:.4f} (iteration {best['iteration']})")
        info(f"Best hypothesis: {best['hypothesis']}")
    else:
        error("No successful runs in this session.")

    info(f"\nFull experiment log saved to: logs/{task_name}_knowledge.json")


def main():
    parser = argparse.ArgumentParser(description="RD-Agent: Autonomous Data Science Loop")
    parser.add_argument("--task",       default="iris", help="Task name: iris | wine")
    parser.add_argument("--iterations", default=5,      type=int, help="Number of R&D iterations")
    parser.add_argument("--api-key",    default=None,   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key found.")
        print("Set it with:  export OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    run_loop(
        task_name=args.task,
        iterations=args.iterations,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
