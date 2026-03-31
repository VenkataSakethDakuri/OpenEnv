"""
Inference Script — Inventory Optimization Environment
=======================================================
Required env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import json
import textwrap

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from server.inventory_env import InventoryEnvironment
from server.constants import EXTRA_INVENTORY_COST
from models import InventoryAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_DAYS = 30


SYSTEM_PROMPT = textwrap.dedent("""
    You are an inventory management AI agent. Each day you receive the current state
    of a retail store with 5 products: electronics, clothing, groceries, furniture, toys.

    Groceries are perishable (5-day shelf life). Other products don't expire.

    Product selling prices: electronics=$150, clothing=$40, groceries=$10, furniture=$200, toys=$25
    Product cost prices: electronics=$100, clothing=$25, groceries=$5, furniture=$130, toys=$12
    Profit margins: electronics=$50, clothing=$15, groceries=$5, furniture=$70, toys=$13
    Shipping costs per unit: slow=$2 (5 days), medium=$5 (3 days), fast=$10 (1 day)
    Warehouse capacity: electronics=100, clothing=200, groceries=500, furniture=50, toys=300

    Events (like black_friday, christmas) boost demand when their countdown hits 0 and last for 2 days.
    Weekends (day%7 == 5 or 6) have 1.2x demand.

    CRITICAL STRATEGY:
    - You MUST restock products when inventory is low. If you don't buy, you run out of
      stock and miss sales. Missed sales = lost revenue = negative reward.
    - Check today's demand to estimate tomorrow's needs.
    - Do NOT overbuy when demand is low — unsold stock ties up cash and perishables expire.
    - Prioritize high-margin products: furniture ($70 profit), electronics ($50 profit).
    - Stock up BEFORE events hit (check event countdowns).

    Each day you must respond with a JSON action:
    {
        "buy_quantities": {"product_name": quantity, ...},
        "delivery_method": "slow" | "medium" | "fast",
        "liquidate": {"product_name": quantity, ...}
    }

    - buy_quantities: products and amounts to order.
    - delivery_method: shipping speed for this order
    - liquidate: products and amounts to dispose of (no revenue, empty {} to skip)
      Use liquidate to free up warehouse space before a restock.

    You will see what demand occurred today AFTER it happened. Use this to spot trends
    and plan restocking. A negative reward means your last action was bad — adjust.

    Do NOT buy more than you can afford. Do NOT buy on the last day.
    Respond with ONLY valid JSON, no explanation.
""").strip()


def format_observation(obs):
    """Convert observation into a readable prompt for the LLM."""

    # format inventory with batch detail, remaining capacity, and extra cost
    inv_lines = []
    for product, batches in obs.updated_inventory.items():
        total = sum(b[0] for b in batches)
        remaining = obs.remaining_capacity.get(product, 0)
        extra_cost = EXTRA_INVENTORY_COST.get(product, 0)
        batch_detail = ", ".join(
            f"{b[0]} units" + (f" ({b[1]}d left)" if b[1] is not None else "")
            for b in batches
        )
        inv_lines.append(f"  {product}: {total} total [{batch_detail}] | space left: {remaining} (extra space: ${extra_cost}/unit)")
    inv_text = "\n".join(inv_lines)

    # format events
    event_lines = []
    for event, days in obs.updated_events.items():
        if days > 0:
            event_lines.append(f"  {event}: in {days} days")
        else:
            event_lines.append(f"  {event}: ACTIVE NOW")
    events_text = "\n".join(event_lines) if event_lines else "  None"

    # format deliveries
    delivery_lines = []
    for delivery in obs.updated_deliveries:
        for product, shipment in delivery.items():
            qty, arrival_day = shipment
            days_away = arrival_day - obs.current_day
            delivery_lines.append(f"  {product}: {qty} units arriving in {days_away} days")
    deliveries_text = "\n".join(delivery_lines) if delivery_lines else "  None"

    # format demand (already happened today — feedback, not prediction)
    demand_lines = []
    for product, units in obs.demand_today.items():
        demand_lines.append(f"  {product}: {units} units")
    demand_text = "\n".join(demand_lines) if demand_lines else "  No demand data yet"

    prompt = f"""Day: {obs.current_day}/{MAX_DAYS}
Cash: ${obs.total_cash:.2f}
Day Profit: ${obs.day_profit:.2f}
Total Profit: ${obs.total_profit:.2f}
Last Step Reward: {obs.reward:.3f}

Inventory:
{inv_text}

Demand That Occurred Today:
{demand_text}

Upcoming Events:
{events_text}

Pending Deliveries:
{deliveries_text}

Respond with your action as JSON."""

    return prompt


def parse_action(response_text):
    """Parse LLM response into InventoryAction."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)
        return InventoryAction(**data)
    except Exception as e:
        print(f"  [DEBUG] Parse FAILED: {e}")
        print(f"  [DEBUG] Raw LLM response: {response_text[:500]}")
        return InventoryAction(
            buy_quantities={},
            delivery_method="slow",
            liquidate={},
        )


def run_task(client, task_name):
    """Run a single task and return total profit."""
    env = InventoryEnvironment(task_name)
    obs = env.reset()

    print(f"\n{'=' * 50}")
    print(f"Task: {task_name.upper()} | Cash: ${obs.total_cash:.2f} | Days: {env.max_days}")
    print(f"{'=' * 50}")

    for day in range(1, env.max_days + 1):
        if obs.done:
            print("Episode ended early.")
            break

        user_prompt = format_observation(obs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=300,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  LLM request failed: {exc}. Skipping turn.")
            response_text = "{}"

        action = parse_action(response_text)

        print(f"Day {day}: buy={action.buy_quantities} delivery={action.delivery_method} liquidate={action.liquidate}")

        obs = env.step(action)

        print(f"  Cash: ${obs.total_cash:.2f} | Day Profit: ${obs.day_profit:.2f} | Reward: {obs.reward:.3f}")

    print(f"Task {task_name} complete | Total profit: ${obs.total_profit:.2f}")
    return obs.total_profit


def main():
    from server.grader import grade_all, compute_baselines

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # print baselines first
    print(f"\n{'=' * 50}")
    print("BASELINES")
    print(f"{'=' * 50}")
    for task_name in ["easy", "medium", "hard"]:
        floor, ceiling = compute_baselines(task_name)
        print(f"  {task_name}: floor=${floor:.2f} (passive) | ceiling=${ceiling:.2f} (heuristic)")

    results = {}
    for task_name in ["easy", "medium", "hard"]:
        profit = run_task(client, task_name)
        results[task_name] = profit

    scores = grade_all(results)

    print(f"\n{'=' * 50}")
    print("FINAL SCORES")
    print(f"{'=' * 50}")
    for task_name, score in scores.items():
        floor, ceiling = compute_baselines(task_name)
        print(f"  {task_name}: {score:.3f} (profit: ${results[task_name]:.2f} | floor: ${floor:.2f} | ceiling: ${ceiling:.2f})")


if __name__ == "__main__":
    main()