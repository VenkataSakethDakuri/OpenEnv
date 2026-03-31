"""
Grader for inventory optimization tasks.
Scores agent performance on a 0.0-1.0 scale using floor/ceiling approach.
  - floor: passive agent (no buys, just sells initial stock until empty)
  - ceiling: heuristic agent (buys to meet average demand each day)
"""

from server.inventory_env import InventoryEnvironment
from models import InventoryAction
from server.constants import TASKS, BASE_PRICES, COST_PRICES, SHIPPING_COST


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
    """Ceiling baseline: smart heuristic that stocks up before events."""
    task = TASKS[task_name]
    env = InventoryEnvironment(task_name)
    obs = env.reset()

    while not obs.done:
        buy = {}
        delivery = "medium"
        liquidate = {}

        # check if any event is imminent (within 3 days)
        event_soon = False
        for event, days in obs.updated_events.items():
            if 0 < days <= 3:
                event_soon = True
                break

        for product, (lo, hi) in task["base_demand"].items():
            avg_demand = (lo + hi) // 2
            current = sum(b[0] for b in obs.updated_inventory.get(product, []))

            if event_soon:
                # stock up 5 days' worth before events, use fast shipping
                target = avg_demand * 5
                delivery = "fast"
            else:
                # normal: keep 3 days' buffer
                target = avg_demand * 3

            if current < target:
                buy[product] = target - current

        # liquidate groceries about to expire (1 day left)
        for batch in obs.updated_inventory.get("groceries", []):
            if batch[1] is not None and batch[1] <= 1:
                liquidate["groceries"] = liquidate.get("groceries", 0) + batch[0]

        # don't buy on last 2 days
        if obs.current_day >= task["max_days"] - 2:
            buy = {}

        # don't buy more than cash allows (rough check)
        total_cost = sum(qty * (COST_PRICES[p] + SHIPPING_COST[delivery]) for p, qty in buy.items())
        if total_cost > obs.total_cash * 0.8:
            # scale down proportionally
            scale = (obs.total_cash * 0.8) / total_cost if total_cost > 0 else 0
            buy = {p: max(1, int(qty * scale)) for p, qty in buy.items()}

        action = InventoryAction(
            buy_quantities=buy,
            delivery_method=delivery,
            liquidate=liquidate,
        )
        obs = env.step(action)

    return obs.total_profit


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