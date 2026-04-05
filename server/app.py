from openenv.core.env_server import create_app
from server.inventory_env import InventoryEnvironment
from server.grader import grade, compute_baselines
from server.constants import TASKS
from models import InventoryAction, InventoryObservation

app = create_app(InventoryEnvironment, InventoryAction, InventoryObservation, env_name="inventory_env")


@app.get("/tasks")
def list_tasks():
    """List available tasks with full schemas."""
    task_list = []
    for name, config in TASKS.items():
        demand = {p: list(v) for p, v in config["base_demand"].items()}
        task_list.append({
            "task_name": name,
            "seed": config["seed"],
            "max_days": config["max_days"],
            "initial_cash": config["initial_cash"],
            "initial_stock": config["initial_stock"],
            "inventory_capacity": config["inventory_capacity"],
            "base_demand": demand,
            "events": config["events"],
        })
    return {"tasks": task_list}


@app.post("/grader")
def grader_endpoint(task_name: str, agent_profit: float):
    """Return the evaluation score for an episode."""
    if task_name not in TASKS:
        return {"error": f"Unknown task: {task_name}. Available: {list(TASKS.keys())}"}
    floor, ceiling = compute_baselines(task_name)
    score = grade(task_name, agent_profit)
    return {
        "task_name": task_name,
        "agent_profit": agent_profit,
        "floor": floor,
        "ceiling": ceiling,
        "score": score,
    }


@app.get("/baseline")
def baseline_endpoint(task_name: str = "easy"):
    """Run baseline inference on a task and return score."""
    import subprocess
    import os
    import re

    if task_name not in TASKS:
        return {"error": f"Unknown task: {task_name}. Available: {list(TASKS.keys())}"}

    env = os.environ.copy()
    env["TASK_NAME"] = task_name

    try:
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )
        output = result.stdout

        # parse score from output
        score = None
        for line in output.splitlines():
            if task_name + ":" in line and "profit" in line:
                score_match = re.search(r"(\d+\.\d+)\s*\(profit", line)
                if score_match:
                    score = float(score_match.group(1))

        return {
            "task_name": task_name,
            "score": score,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out (20 min limit)"}
    except Exception as e:
        return {"error": str(e)}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()