"""
Grader for inventory optimization tasks.
Scores agent performance on a 0.0-1.0 scale using floor/ceiling approach.
  - floor: passive agent (no buys, just sells initial stock until empty)
  - ceiling: theoretical max profit with perfect demand knowledge
"""

from server.inventory_env import InventoryEnvironment
from models import InventoryAction
from server.constants import (
    TASKS, BASE_PRICES, COST_PRICES, SHIPPING_COST, EVENT_EFFECTS,
    WEEKEND_MULTIPLIER, EVENT_DURATION,
)

import random


def _run_passive(task_name):
    """Floor baseline: do nothing, just sell whatever initial stock covers."""
    env = InventoryEnvironment(task_name)
    obs = env.reset()

    while not obs.done:
        action = InventoryAction(
            buy_quantities={},
            delivery_method="slow",
            liquidate={},
        )
        obs = env.step(action)

    return obs.total_profit


def _run_heuristic(task_name):
    task = TASKS[task_name]
    events = dict(task["events"])

    total_demand = {p: 0 for p in task["base_demand"]}

    for day in range(1, task["max_days"] + 1):
        # tick events
        for event_name in events:
            events[event_name] -= 1

        rng = random.Random(task["seed"] * 1000 + day)

        for product, (lo, hi) in task["base_demand"].items():
            demand = rng.randint(lo, hi)

            # weekend boost
            if day % 7 == 5 or day % 7 == 6:
                demand = int(WEEKEND_MULTIPLIER * demand)

            # event multipliers
            for event_name, days_left in events.items():
                if -EVENT_DURATION < days_left <= 0 and event_name in EVENT_EFFECTS:
                    mult = EVENT_EFFECTS[event_name].get(product, 1.0)
                    demand = int(demand * mult)

            total_demand[product] += demand

    total_profit = 0.0

    # sell the initial stock first
    initial_stock = task["initial_stock"]

    for product in task["base_demand"]:
        total_profit += min(initial_stock.get(product, 0), total_demand[product]) * BASE_PRICES[product]
        total_demand[product] = max(0, total_demand[product] - initial_stock.get(product, 0))

        # cost price and shipping cost applies after initial stock
        total_profit += total_demand[product] * (BASE_PRICES[product] - COST_PRICES[product] - SHIPPING_COST["slow"])

    return total_profit


def compute_baselines(task_name):
    """Pre-compute floor and ceiling for a task."""
    floor = _run_passive(task_name)
    ceiling = _run_heuristic(task_name)
    return floor, ceiling


def grade(task_name, agent_profit):
    """
    Grade agent performance on 0.0-1.0 scale.

    Args:
        task_name: "easy", "medium", or "hard"
        agent_profit: total profit achieved by the agent

    Returns:
        float score between 0.0 and 1.0
    """
    floor, ceiling = compute_baselines(task_name)

    if ceiling <= floor:
        return 1.0 if agent_profit >= ceiling else 0.0

    score = (agent_profit - floor) / (ceiling - floor)
    return max(0.0, min(1.0, score))


def grade_all(results):
    """
    Grade all 3 tasks.

    Args:
        results: dict of {task_name: agent_profit}

    Returns:
        dict of {task_name: score}
    """
    scores = {}
    for task_name, agent_profit in results.items():
        scores[task_name] = grade(task_name, agent_profit)
    return scores


if __name__ == "__main__":
    print("Computing baselines for all tasks...")
    for task_name in ["easy", "medium", "hard"]:
        floor, ceiling = compute_baselines(task_name)
        print(f"  {task_name}: floor={floor:.2f}, ceiling={ceiling:.2f}")