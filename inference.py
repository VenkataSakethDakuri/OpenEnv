"""
Inference Script - Inventory Optimization Environment
=====================================================
Required env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Hugging Face token (preferred for HF Router).

Supported key env vars (first non-empty wins): HF_TOKEN, API_KEY, OPENAI_API_KEY.
For non-OpenAI endpoints, a dummy key is used when no key is provided because
the OpenAI Python SDK requires a non-empty api_key argument.
"""

import os
import json
import textwrap

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from server.inventory_env import InventoryEnvironment
from server.constants import EXTRA_INVENTORY_COST, EVENT_DURATION
from models import InventoryAction

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen3-32B"
TASK_NAME = os.getenv("TASK_NAME") or "easy"
MAX_DAYS = 30

SYSTEM_PROMPT = textwrap.dedent("""
    You are an inventory management AI agent. Each day you receive the current state
    of a retail store with 5 products: electronics, clothing, groceries, furniture, toys.

    You will be shown your decision history from recent days so you can learn from
    past outcomes. Use this history to spot demand trends, identify what worked vs.
    what didn't, and adjust your strategy accordingly.

    Groceries are perishable (5-day shelf life). Other products don't expire.

    Product selling prices: electronics=$150, clothing=$40, groceries=$10, furniture=$200, toys=$25
    Product cost prices: electronics=$100, clothing=$25, groceries=$5, furniture=$130, toys=$12
    Profit margins: electronics=$50, clothing=$15, groceries=$5, furniture=$70, toys=$13
    Shipping costs per unit: slow=$2 (3-7 days), medium=$5 (2-4 days), fast=$10 (1 day, always reliable)
    Warehouse capacity: electronics=100, clothing=200, groceries=500, furniture=50, toys=300

    Events (like black_friday, christmas) boost demand when their countdown hits 0 and last for 2 days.
    Weekends (day%7 == 5 or 6) have 1.2x demand.

    CRITICAL STRATEGY:
    - Review your history: if reward was negative, identify why and change approach.
    - Track demand trends across days.
    - You MUST restock products when inventory is low. Missed sales = lost revenue = negative reward.
    - Do NOT overbuy when demand is low — unsold stock ties up cash and perishables expire.
    - Stock up BEFORE events hit (check event countdowns — order 3-5 days ahead).
    - When no events are approaching, slow shipping is often sufficient and saves significant cost.
    - Near end of episode (last 2 days), stop buying — focus on selling remaining stock.

    DYNAMIC PRICING:
    You can set a price multiplier (0.5 to 1.5) per product each day. Default is 1.0.
    - Lower price (e.g. 0.7) = more demand but less revenue per unit. Good for clearing excess stock.
    - Higher price (e.g. 1.3) = less demand but more revenue per unit. Good when stock is low.
    - Price elasticity varies across different products.
    - Elasticity values: electronics=1.2, clothing=1.5, groceries=0.4, furniture=0.8, toys=1.3

    Each day you must respond with a JSON action:
    {
        "buy_quantities": {"product_name": quantity, ...},
        "delivery_method": "slow" | "medium" | "fast",
        "liquidate": {"product_name": quantity, ...},
        "price_multipliers": {"product_name": multiplier, ...}
    }

    - buy_quantities: products and amounts to order.
    - delivery_method: shipping speed for this order
    - liquidate: products and amounts to dispose of (no revenue, empty {} to skip)
      Use liquidate to free up warehouse space before a restock.
    - price_multipliers: set selling price multiplier per product (0.5-1.5, default 1.0 if omitted)

    LEARNING FROM HISTORY:
    - Compare your past buy quantities to the demand that followed — were you over or under?
    - If you see repeated stockouts for a product, increase orders for it.
    - If groceries expired, you overbought — reduce grocery orders or use faster shipping.
    - A negative reward means your last action was bad — adjust immediately.

    Before responding with JSON, briefly reason (2-3 lines max):
    1. What did I learn from recent history? What went wrong/right?
    2. What products need restocking vs. are overstocked?
    3. Are any events approaching?

    Then output ONLY the final JSON action on the last line.
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
        elif -EVENT_DURATION < days <= 0:
            event_lines.append(f"  {event}: ACTIVE NOW")
        else:
            event_lines.append(f"  {event}: ended")
    events_text = "\n".join(event_lines) if event_lines else "  None"

    # format deliveries
    delivery_lines = []
    for delivery in obs.updated_deliveries:
        for product, shipment in delivery.items():
            qty, arrival_day = shipment
            days_away = arrival_day - obs.current_day
            delivery_lines.append(f"  {product}: {qty} units arriving in {days_away} days")
    deliveries_text = "\n".join(delivery_lines) if delivery_lines else "  None"

    # format demand (yesterday's demand — feedback, not prediction)
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

Yesterday's Demand:
{demand_text}

Upcoming Events:
{events_text}

Pending Deliveries:
{deliveries_text}

Respond with your action as JSON."""

    return prompt


def parse_action(response_text):
    """Parse LLM response into InventoryAction. Extracts JSON even if surrounded by text."""
    try:
        text = response_text.strip()

        # strip markdown code fences
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # find the first { and last } to extract JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

        data = json.loads(text)

        # only keep valid fields
        clean = {}
        if "buy_quantities" in data:
            clean["buy_quantities"] = data["buy_quantities"]
        if "delivery_method" in data:
            clean["delivery_method"] = data["delivery_method"]
        if "liquidate" in data:
            clean["liquidate"] = data["liquidate"]
        if "price_multipliers" in data:
            clean["price_multipliers"] = data["price_multipliers"]

        return InventoryAction(**clean)
    except Exception as e:
        print(f"  [DEBUG] Parse FAILED: {e}")
        print(f"  [DEBUG] Raw LLM response: {response_text[:500]}")
        return InventoryAction(
            buy_quantities={},
            delivery_method="slow",
            liquidate={},
            price_multipliers={},
        )        


HISTORY_WINDOW = 7  # rolling window of past days to include in context


def run_task(client, task_name):
    """Run a single task and return total profit."""
    env = InventoryEnvironment(task_name)
    obs = env.reset()

    print(f"\n{'=' * 50}")
    print(f"Task: {task_name.upper()} | Cash: ${obs.total_cash:.2f} | Days: {env.max_days}")
    print(f"{'=' * 50}")

    # Rolling history of (user_observation, assistant_response) pairs
    history = []

    for day in range(1, env.max_days + 1):
        if obs.done:
            print("Episode ended early.")
            break

        user_prompt = format_observation(obs)

        # Build messages: system + history context + current observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        recent = history[-HISTORY_WINDOW:]
        if recent:
            # Tell the LLM it's about to see its past decisions and their outcomes
            messages.append({
                "role": "user",
                "content": f"Here is your decision history from the last {len(recent)} day(s). "
                           "Use this to identify demand trends, adjust restocking, and avoid repeating mistakes.",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I'll review my past decisions and their outcomes to make better choices today.",
            })
            for past_user, past_assistant in recent:
                messages.append({"role": "user", "content": past_user})
                messages.append({"role": "assistant", "content": past_assistant})

        messages.append({"role": "user", "content": user_prompt})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=500,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  LLM request failed: {exc}. Skipping turn.")
            response_text = "{}"

        # Save this turn to rolling history
        history.append((user_prompt, response_text))

        action = parse_action(response_text)

        print(f"Day {day}: buy={action.buy_quantities} delivery={action.delivery_method} liquidate={action.liquidate} prices={action.price_multipliers}")

        obs = env.step(action)

        print(f"  Cash: ${obs.total_cash:.2f} | Day Profit: ${obs.day_profit:.2f} | Reward: {obs.reward:.3f}")

    print(f"Task {task_name} complete | Total profit: ${obs.total_profit:.2f}")
    return obs.total_profit


def main():
    from server.grader import grade, compute_baselines

    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is not set. Please export MODEL_NAME before running inference.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # print baseline for selected task
    floor, ceiling = compute_baselines(TASK_NAME)
    print(f"\n{'=' * 50}")
    print(f"BASELINE ({TASK_NAME}): floor=${floor:.2f} (passive) | ceiling=${ceiling:.2f} (heuristic)")
    print(f"{'=' * 50}")

    profit = run_task(client, TASK_NAME)
    score = grade(TASK_NAME, profit)

    print(f"\n{'=' * 50}")
    print("FINAL SCORE")
    print(f"{'=' * 50}")
    print(f"  {TASK_NAME}: {score:.3f} (profit: ${profit:.2f} | floor: ${floor:.2f} | ceiling: ${ceiling:.2f})")


if __name__ == "__main__":
    main()