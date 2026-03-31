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

    # track recent demand to adapt ordering
    demand_history = {}

    while not obs.done:
        buy = {}
        liquidate = {}

        # determine nearest event distance
        nearest_event_days = 999
        for event, days in obs.updated_events.items():
            if 0 < days < nearest_event_days:
                nearest_event_days = days

        # pick shipping based on urgency
        if nearest_event_days <= 2:
            delivery = "fast"
        elif nearest_event_days <= 5:
            delivery = "medium"
        else:
            delivery = "slow"

        # update demand history from observation
        if obs.demand_today:
            for product, units in obs.demand_today.items():
                if product not in demand_history:
                    demand_history[product] = []
                demand_history[product].append(units)

        for product, (lo, hi) in task["base_demand"].items():
            avg_demand = (lo + hi) // 2

            # use recent demand if available (last 5 days)
            if product in demand_history and len(demand_history[product]) >= 2:
                recent = demand_history[product][-5:]
                avg_demand = max(avg_demand, int(sum(recent) / len(recent)))

            current = sum(b[0] for b in obs.updated_inventory.get(product, []))

            # count in-transit units
            in_transit = 0
            for d in obs.updated_deliveries:
                for p, shipment in d.items():
                    if p == product:
                        in_transit += shipment[0]

            available = current + in_transit

            # how many days of stock to target
            if nearest_event_days <= 5:
                target = avg_demand * 6
            else:
                target = avg_demand * 4

            # prioritize high-margin products — order more aggressively
            margin = BASE_PRICES[product] - COST_PRICES[product]
            if margin >= 50:  # electronics, furniture
                target = int(target * 1.3)

            if available < target:
                buy[product] = target - available

        # liquidate groceries about to expire (1 day left)
        for batch in obs.updated_inventory.get("groceries", []):
            if batch[1] is not None and batch[1] <= 1:
                liquidate["groceries"] = liquidate.get("groceries", 0) + batch[0]

        # stop buying when deliveries can't arrive in time
        days_left = task["max_days"] - obs.current_day
        if delivery == "slow" and days_left <= 5:
            buy = {}
        elif delivery == "medium" and days_left <= 3:
            buy = {}
        elif delivery == "fast" and days_left <= 1:
            buy = {}

        # don't buy more than cash allows (rough check)
        total_cost = sum(qty * (COST_PRICES[p] + SHIPPING_COST[delivery]) for p, qty in buy.items())
        if total_cost > obs.total_cash * 0.85:
            scale = (obs.total_cash * 0.85) / total_cost if total_cost > 0 else 0
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