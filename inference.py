"""
Inference Script - Long-Horizon Inventory Optimization
======================================================
Required env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Hugging Face token (preferred for HF Router).
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

SYSTEM_PROMPT = textwrap.dedent("""
    You are a long-horizon inventory planning AI. You manage a retail store for 90 days.

    CRITICAL: This is a MEMORY TEST. You receive corporate directives throughout the quarter.
    Directives are shown in FULL only ONCE when issued. After that, you only see their IDs.
    You MUST remember and comply with ALL active directives or face penalties.

    You have two memory fields that persist between steps:
    - "notes_to_self": Shown back to you next turn. Use however you see fit.
    - "weekly_plan": Persistent until you overwrite it.

    PRODUCTS (5):
    Product      | Sell  | Cost  | Margin | Shelf Life
    electronics  | $150  | $100  | $50    | no expiry
    clothing     | $40   | $25   | $15    | no expiry
    groceries    | $10   | $5    | $5     | 5 days
    furniture    | $200  | $130  | $70    | no expiry
    toys         | $25   | $12   | $13    | no expiry

    SHIPPING: slow=$2/unit (3-7 days), medium=$5/unit (2-4 days), fast=$10/unit (1 day)
    You can choose a DIFFERENT shipping speed per product.
    WAREHOUSE: electronics=100, clothing=200, groceries=500, furniture=50, toys=300
    OVERAGE: Ordering beyond capacity costs extra per unit: electronics=$20, clothing=$5, groceries=$2, furniture=$30, toys=$4
    PRICE ELASTICITY: electronics=1.2, clothing=1.5, groceries=0.4, furniture=0.8, toys=1.3

    LIQUIDATE: Dispose of stock for no revenue. Units are removed from inventory.
    Use to: free warehouse space, comply with recall directives, dump expiring groceries.

    LOANS (recovery from mistakes):
    - If cash drops below $100, you can set "take_loan": true to borrow $500.
    - Interest: 3% daily compound on outstanding balance.
    - Auto-repayment: 15% of daily revenue goes toward the loan.
    - At episode end, remaining balance is subtracted from total profit.
    - Maximum 2 loans per episode. After that, going broke = game over.
    - Strategy: take a loan early if needed (more time to repay), but avoid if possible.

    Events boost demand when countdown hits 0 (last 3 days).
    Weekends (day%7 == 5 or 6) have 1.2x demand.

    DIRECTIVES (the key challenge):
    - "new_directives": Full text shown ONCE on arrival day. READ CAREFULLY.
    - "active_directive_ids": Just IDs of rules you must follow. No text reminder.
    - "directive_violations_last_step": Shows which rules you broke last step.
    - Some directives MODIFY previous ones. Track which version is current!

    MILESTONES:
    - Shown in "milestones" field with target, deadline, and current progress.
    - Big bonus reward when achieved.
    - Plan ahead to hit them on time.

    Your performance is evaluated on how well you manage the store over the full
    90-day horizon. Directive compliance, profitability, and efficient planning
    all contribute to your score.

    RESPOND WITH ONLY A SINGLE JSON OBJECT. No reasoning, no explanation, no markdown fences.
    Output MUST start with { and end with }. Nothing else.

    JSON format:
    {
        "buy_quantities": {"electronics": 0, "clothing": 0, "groceries": 0, "furniture": 0, "toys": 0},
        "delivery_methods": {"electronics": "slow", "clothing": "slow", "groceries": "fast", "furniture": "slow", "toys": "slow"},
        "liquidate": {"electronics": 0, "clothing": 0, "groceries": 0, "furniture": 0, "toys": 0},
        "price_multipliers": {"electronics": 1.0, "clothing": 1.0, "groceries": 1.0, "furniture": 1.0, "toys": 1.0},
        "notes_to_self": "Track directives, violations, plans here. This is your ONLY memory between steps.",
        "weekly_plan": "Your strategic plan for the current week.",
        "take_loan": false
    }

    delivery_methods must be exactly one of: "slow", "medium", "fast"
    price_multipliers must be between 0.5 and 1.5
    Use notes_to_self to remember directive text, track violations, and plan ahead.
""").strip()


def format_observation(obs):
    """Convert observation into a prompt for the LLM."""

    # Inventory
    inv_lines = []
    for product, batches in obs.updated_inventory.items():
        total = sum(b[0] for b in batches)
        remaining = obs.remaining_capacity.get(product, 0)
        batch_detail = ", ".join(
            f"{b[0]}u" + (f"({b[1]}d)" if b[1] is not None else "")
            for b in batches
        )
        inv_lines.append(f"  {product}: {total} [{batch_detail}] space:{remaining}")
    inv_text = "\n".join(inv_lines)

    # Events
    event_lines = []
    for event, days in obs.updated_events.items():
        if days > 0:
            event_lines.append(f"  {event}: in {days} days")
        elif -EVENT_DURATION < days <= 0:
            event_lines.append(f"  {event}: ACTIVE")
        else:
            event_lines.append(f"  {event}: ended")
    events_text = "\n".join(event_lines) if event_lines else "  None"

    # Deliveries
    delivery_lines = []
    for delivery in obs.updated_deliveries:
        for product, shipment in delivery.items():
            qty, arrival_day = shipment
            days_away = arrival_day - obs.current_day
            delivery_lines.append(f"  {product}: {qty}u in {days_away}d")
    deliveries_text = "\n".join(delivery_lines) if delivery_lines else "  None"

    # Demand
    demand_lines = []
    for product, units in obs.demand_today.items():
        demand_lines.append(f"  {product}: {units}")
    demand_text = "\n".join(demand_lines) if demand_lines else "  No data yet"

    # Directives
    directive_text = ""
    if obs.new_directives:
        directive_text += "\n*** NEW DIRECTIVES (read carefully, shown ONCE) ***\n"
        for d in obs.new_directives:
            line = f"  [{d['id']}] ({d['type']}): {d['text']}"
            if d.get('expires_day'):
                line += f" [expires day {d['expires_day']}]"
            if d.get('replaces'):
                line += f" [replaces {d['replaces']}]"
            directive_text += line + "\n"

    active_ids = ", ".join(obs.active_directive_ids) if obs.active_directive_ids else "None"

    violations_text = ""
    if obs.directive_violations_last_step:
        violations_text = "\n!!! VIOLATIONS LAST STEP !!!\n"
        for v in obs.directive_violations_last_step:
            violations_text += f"  {v['id']}: {v['text']} (penalty: {v['penalty']})\n"

    # Milestones
    milestone_lines = []
    for name, m in obs.milestones.items():
        status = "DONE" if m["achieved"] else f"current={m['current']:.1f}"
        milestone_lines.append(f"  {name}: target={m['target']} by day {m['deadline']} [{status}]")
    milestones_text = "\n".join(milestone_lines) if milestone_lines else "  None"

    prompt = f"""Day {obs.current_day}/{obs.total_days} | Cash: ${obs.total_cash:.0f} | Day Profit: ${obs.day_profit:.0f} | Total Profit: ${obs.total_profit:.0f} | Reward: {obs.reward:.2f}
{directive_text}{violations_text}
Active Directives: [{active_ids}]

Inventory:
{inv_text}

Last Demand:
{demand_text}

Events:
{events_text}

Deliveries:
{deliveries_text}

Milestones:
{milestones_text}

Loan: balance=${obs.loan_balance:.0f} | taken={obs.loans_taken} | remaining={obs.loans_remaining}

Your Notes: {obs.agent_notes if obs.agent_notes else '(empty)'}
Your Plan: {obs.agent_weekly_plan if obs.agent_weekly_plan else '(empty)'}

"""

    return prompt


def parse_action(response_text):
    """Parse LLM response into InventoryAction."""
    try:
        text = response_text.strip()

        # Strip Qwen3 thinking tokens if present
        if "<think>" in text:
            think_end = text.find("</think>")
            if think_end != -1:
                text = text[think_end + 8:].strip()
            else:
                # Unterminated thinking — take everything after <think>
                text = text[text.find("<think>") + 7:].strip()

        # Strip markdown fences
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # Find the JSON object by matching braces (handles nested dicts)
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")

        # Walk forward counting braces to find matching close
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            # Truncated JSON — try to salvage
            text = text[start:]
            salvaged = _salvage_truncated_json(text)
            if salvaged:
                data = salvaged
            else:
                raise ValueError("Truncated JSON could not be salvaged")
        else:
            text = text[start:end + 1]
            data = json.loads(text, strict=False)

        clean = {}
        for key in ["buy_quantities", "delivery_methods", "liquidate",
                     "price_multipliers", "notes_to_self", "weekly_plan",
                     "take_loan"]:
            if key in data:
                clean[key] = data[key]

        # Fix invalid delivery method values
        if "delivery_methods" in clean and isinstance(clean["delivery_methods"], dict):
            valid_methods = {"slow", "medium", "fast"}
            clean["delivery_methods"] = {
                k: str(v).lower().strip() if str(v).lower().strip() in valid_methods else "slow"
                for k, v in clean["delivery_methods"].items()
                if isinstance(k, str)
            }

        return InventoryAction(**clean)
    except Exception as e:
        print(f"  [DEBUG] Parse FAILED: {e}")
        print(f"  [DEBUG] Raw: {response_text[:300]}")
        return InventoryAction()


def _salvage_truncated_json(text):
    """Try to extract whatever complete key-value pairs exist in truncated JSON."""
    import re

    # Try progressively removing trailing content until it parses
    # Find positions of all complete "key": value pairs
    result = {}

    # Extract complete top-level string values: "key": "value"
    for m in re.finditer(r'"(buy_quantities|delivery_methods|liquidate|price_multipliers|notes_to_self|weekly_plan)"\s*:\s*', text):
        key = m.group(1)
        rest = text[m.end():]

        if rest.startswith('"'):
            # String value — find closing quote
            str_end = rest.find('"', 1)
            if str_end != -1:
                result[key] = rest[1:str_end]
        elif rest.startswith('{'):
            # Dict value — find matching brace
            depth = 0
            for i, c in enumerate(rest):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            result[key] = json.loads(rest[:i+1], strict=False)
                        except json.JSONDecodeError:
                            pass
                        break
        elif rest.startswith('null'):
            result[key] = None

    return result if result else None


def run_task(client, task_name):
    """Run a single task and return total profit."""
    env = InventoryEnvironment(task_name)
    obs = env.reset()

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    print(f"[START] task={task_name} env=quartermaster-env model={MODEL_NAME}", flush=True)

    try:
        for day in range(1, env.max_days + 1):
            if obs.done:
                break

            user_prompt = format_observation(obs)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            error = None
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_completion_tokens=800,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                error = str(exc)
                response_text = "{}"

            action = parse_action(response_text)
            action_str = json.dumps({
                "buy": action.buy_quantities,
                "deliver": action.delivery_methods,
                "liquidate": action.liquidate,
                "prices": action.price_multipliers,
                "notes": action.notes_to_self[:50] if action.notes_to_self else "",
            })

            obs = env.step(action)

            reward = obs.reward
            done = obs.done
            rewards.append(reward)
            steps_taken = day

            print(f"[STEP] step={day} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

            if done:
                break

        from server.grader import grade
        score = grade(task_name, obs.total_profit)
        success = score >= 0.1

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

    return obs.total_profit


def main():
    from server.grader import grade, compute_baselines

    if not MODEL_NAME:
        raise RuntimeError("MODEL_NAME is not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["easy", "medium", "hard"]

    print(f"\n{'=' * 50}")
    print("BASELINES")
    print(f"{'=' * 50}")
    for task_name in tasks:
        floor, ceiling = compute_baselines(task_name)
        print(f"  {task_name}: floor=${floor:.2f} | ceiling=${ceiling:.2f}")

    results = {}
    for task_name in tasks:
        profit = run_task(client, task_name)
        results[task_name] = profit

    print(f"\n{'=' * 50}")
    print("FINAL SCORES")
    print(f"{'=' * 50}")
    for task_name in tasks:
        floor, ceiling = compute_baselines(task_name)
        score = grade(task_name, results[task_name])
        print(f"  {task_name}: {score:.3f} (profit: ${results[task_name]:.2f} | floor: ${floor:.2f} | ceiling: ${ceiling:.2f})")


if __name__ == "__main__":
    main()