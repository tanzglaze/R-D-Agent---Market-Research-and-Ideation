# 🤖 RD-Agent: Autonomous Data Science Research Loop

A portfolio project inspired by Microsoft's RD-Agent. This system uses LLMs to autonomously run a **Research → Develop → Evaluate → Iterate** loop on Kaggle-style data science tasks.

## Architecture

```
Research Agent  →  Dev Agent  →  Executor  →  Evaluator
      ↑                                            |
      └──────────── Knowledge Store ←──────────────┘
```

## Features

- 🔬 **Research Agent** — proposes ML hypotheses (features, models, preprocessing ideas)
- 🛠️ **Dev Agent** — writes executable Python ML code from a hypothesis
- 🏃 **Executor** — runs code in a sandboxed subprocess, captures stdout/stderr
- 📊 **Evaluator** — parses accuracy/score from output, decides if it's an improvement
- 🔁 **Loop Orchestrator** — iterates N times, building on past knowledge
- 📝 **Knowledge Store** — JSON log of all experiments, scores, and learnings

## Setup

```bash
pip install openai scikit-learn pandas numpy
export OPENAI_API_KEY=your_key_here
```

## Run

```bash
# Run the full loop for 5 iterations on the Iris dataset
python loop.py --task iris --iterations 5

# Run on your own CSV
python loop.py --task custom --data path/to/data.csv --target target_column --iterations 3
```

## Project Structure

```
rd_agent/
├── loop.py              # Main orchestrator
├── agents/
│   ├── research_agent.py  # Hypothesis proposer
│   └── dev_agent.py       # Code writer
├── core/
│   ├── executor.py        # Safe code runner
│   ├── evaluator.py       # Score parser
│   └── knowledge_store.py # Experiment logger
├── tasks/
│   └── iris_task.py       # Sample task definition
└── logs/                  # Experiment traces
```
