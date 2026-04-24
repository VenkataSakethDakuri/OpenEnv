---
title: Quartermaster
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Quartermaster — Long-Horizon Inventory RL Environment

A 90-step OpenEnv RL environment that tests **long-horizon planning, memory management, and strategic reasoning** under evolving corporate directives. The agent manages a retail store for a full quarter (90 days) while tracking, remembering, and complying with directives that are shown only once.

**Theme:** Long-Horizon Planning & Instruction Following
**Sub-theme:** Scale AI (business workflow) + Mercor (reward scales with reasoning quality)

## Why Inventory Management?

Retail inventory optimization is a real-world task performed daily by store managers, warehouse operators, and supply chain planners. The agent faces the same challenges as a human manager: uncertain demand, perishable goods, shipping delays, seasonal events, budget constraints, and evolving corporate policies. This environment pushes beyond shallow next-token reasoning by requiring the agent to maintain durable internal representations across 90 steps — well beyond typical context limits for small models.

## What Makes This Environment Challenging

| Challenge | How It Works |
|-----------|-------------|
| **Memory beyond context** | Directives shown in full ONCE, then only IDs. Agent must externalize memory via notes. |
| **Conflicting directives** | Some active rules are mathematically impossible to satisfy simultaneously. Agent must reason about penalty tradeoffs and choose which rule to violate. |
| **Deceptive directives** | "Stock 200 groceries!" looks profitable but triggers cascading waste penalties. Agent must reason about second-order effects. |
| **Directive modifications** | Later directives UPDATE or CANCEL earlier ones. Agent must track which version is current. |
| **90-step horizon** | Decisions on day 5 affect outcomes on day 40. Shipping delays, expiry, and milestone deadlines force genuine long-range planning. |
| **Content-aware planning score** | Notes are scored for quality (directive tracking, evolution, situational awareness) — not just length. Copy-paste is penalized. |

## Environment Overview

Manage 5 products over 90 days with evolving constraints:

| Product | Sell | Cost | Margin | Shelf Life |
|---------|------|------|--------|------------|
| Electronics | $150 | $100 | $50 | No expiry |
| Clothing | $40 | $25 | $15 | No expiry |
| Groceries | $10 | $5 | $5 | 5 days |
| Furniture | $200 | $130 | $70 | No expiry |
| Toys | $25 | $12 | $13 | No expiry |

Each day the agent receives the current store state and must decide:
- **What to buy** and how much of each product
- **How to ship** — slow (cheap but unreliable), medium, or fast (expensive but guaranteed)
- **What to liquidate** — dispose of expiring or excess stock
- **How to price** — set per-product price multipliers that affect demand via elasticity
- **What to remember** — write notes and plans that persist between steps

## Environment Design Highlights

### Batch-Tracked Inventory with FIFO

Inventory is tracked per batch with individual expiry dates. Groceries expire after 5 days. Selling and liquidation follow FIFO (First In, First Out):

```json
{"groceries": [[20, 3], [15, 5], [10, 1]]}
```
Three batches: 20 units (3 days left), 15 units (5 days left), 10 units (1 day left — liquidate or lose them).

### Dynamic Pricing with Price Elasticity

The agent can set per-product price multipliers (0.5x to 1.5x). Demand responds via realistic elasticity:

| Product | Elasticity | Effect of 1.3x price |
|---------|-----------|----------------------|
| Electronics | 1.2 | Demand drops ~24% |
| Clothing | 1.5 | Demand drops ~38% |
| Groceries | 0.4 | Demand drops only ~11% |
| Furniture | 0.8 | Demand drops ~22% |
| Toys | 1.3 | Demand drops ~33% |

### Delivery Jitter

Slow delivery has +/-2 day variance, medium has +/-1 day. Only fast delivery (at 5x cost) is guaranteed next-day. The agent must account for uncertainty when planning restocks before events.

### Seasonal Events with Demand Spikes

Events trigger 3-day demand multipliers (up to 3.5x during Black Friday). The agent sees countdowns and must stock up in advance. Some events (supply disruptions, competitor launches) actually *reduce* demand — testing whether the agent adapts.

### Directive Memory System

The core innovation: corporate directives arrive throughout the quarter and are **shown in full only once**. After the arrival day, the agent only sees a list of active directive IDs. This forces the agent to use `notes_to_self` to externalize memory — a genuine test of long-horizon instruction following beyond context limits.

Directive types include: minimum stock rules, budget caps, shipping restrictions, price ranges, forced liquidations, order freezes, waste limits, and cash reserve policies.

### Conflicting Directives (Strategic Violation)

Some directive pairs are **mathematically impossible** to satisfy simultaneously:

- Order freeze ("no purchases allowed") active alongside min_stock ("maintain 80+ electronics")
- Agent must calculate which violation has the lower penalty and choose strategically
- Tests reasoning ability that frontier models struggle with: intentionally violating rules

### Deceptive Directives (Second-Order Reasoning)

Some directives **appear beneficial but are traps**:

- "URGENT: Stock 200 groceries for summer push!" (penalty only -0.5 if ignored)
- But: groceries expire in 5 days, triggering waste limit violations at -1.0/step
- Smart agent: ignores the directive (total cost -0.5) rather than triggering a waste cascade (total cost -3.0+)

### Milestone System

Time-bound targets that reward forward planning:
- "Reach $2000 profit by day 50" → +2.0 bonus
- "Zero grocery waste for 14 consecutive days" → +4.0 bonus
- "Stock 100+ toys by day 79" → +3.0 bonus

Milestones require advance preparation and cannot be achieved reactively on the deadline day.

## Action Space

```python
class InventoryAction(Action):
    buy_quantities: Dict[str, int] = {}          # What to order
    delivery_methods: Dict[str, Literal["slow", "medium", "fast"]] = {}  # Per-product shipping
    liquidate: Dict[str, int] = {}               # Dispose of stock (no revenue)
    price_multipliers: Dict[str, float] = {}     # Per-product pricing (0.5-1.5x)
    notes_to_self: str = ""                      # Agent's private scratchpad (persists)
    weekly_plan: Optional[str] = None            # Persistent plan (until overwritten)
```

| Field | Description |
|-------|-------------|
| `buy_quantities` | Products and amounts to order. Empty `{}` to skip buying. |
| `delivery_methods` | Per-product shipping speed. `"slow"` ($2/unit, 3-7 days), `"medium"` ($5/unit, 2-4 days), `"fast"` ($10/unit, 1 day). Defaults to `"slow"` if omitted. |
| `liquidate` | Products and amounts to dispose of (no revenue). Use for expiring groceries or complying with recall directives. |
| `price_multipliers` | Per-product selling price multiplier (0.5-1.5). Affects demand via elasticity. Default 1.0 if omitted. |
| `notes_to_self` | Agent's private scratchpad. Persisted and returned in next observation. |
| `weekly_plan` | Persistent plan shown every step until overwritten. |

## Observation Space

```python
class InventoryObservation(Observation):
    current_day: int
    total_days: int
    total_cash: float
    day_profit: float
    total_profit: float
    demand_today: Dict[str, int]                    # Yesterday's realized demand
    updated_inventory: Dict[str, List[List[Optional[int]]]]  # [[qty, days_left], ...]
    remaining_capacity: Dict[str, int]              # Warehouse space per product
    updated_events: Dict[str, int]                  # Countdowns (negative = active)
    updated_deliveries: List[Dict]                  # In-transit shipments
    new_directives: List[Dict]                      # Full text, shown ONCE on arrival
    active_directive_ids: List[str]                  # Only IDs after arrival day
    directive_violations_last_step: List[Dict]       # Which rules were broken
    milestones: Dict[str, Dict]                     # Target, deadline, progress
    agent_notes: str                                # Returned from previous step
    agent_weekly_plan: str                          # Persistent plan
```

## Reward Structure

### Dense Per-Step Signals (weighted sum, range [-1, +1])

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| R_directives | 40% | Compliance with active rules |
| R_planning | 20% | Quality of agent's planning notes (content-aware scoring) |
| R_revenue | 15% | Revenue captured vs maximum possible |
| R_fulfillment | 15% | Demand met (units sold / units demanded) |
| R_waste | 10% | Spoilage and liquidation efficiency |

### R_planning: Advanced Content-Aware Scoring

Unlike simple length checks, planning notes are scored on **5 content signals**:

1. **Directive tracking (+0.50)** — Does the agent mention active directive IDs in notes? Scored proportionally to how many active IDs are referenced.
2. **Situational awareness (+0.30)** — References to product names, quantities, and events indicate the agent is reasoning about current state.
3. **Note evolution (+0.30)** — Exact copy-paste from previous step is penalized (-0.30). Evolving notes that adapt to new information are rewarded.
4. **Violation acknowledgment (+0.20)** — When violations occur, does the agent acknowledge them and plan corrections?
5. **Plan structure (+0.20)** — Structured plans with bullet points, numbers, and actionable items score higher.

**Range:** Empty notes/plan = -1.0. Perfect notes tracking all directives with evolving structured plans = +1.0. This directly rewards frontier models that produce more detailed, higher-quality reasoning (Mercor sub-theme alignment: reward scales with token output quality).

### Sparse Signals (event-driven)

- **Milestone bonuses**: +1.5 to +5.0 for achieving targets by deadlines
- **Directive violations**: -0.3 to -5.0 per violated rule per step
- **Hard fail gates**: -1.0 (unaffordable order attempted), -2.0 (bankruptcy, cash < $10), -1.0 (idle 3+ consecutive days)

## Directive System

Directives arrive throughout the 90-day episode and test long-term memory:

| Type | Examples | Penalty Range |
|------|----------|---------------|
| `min_stock` | "Maintain 50+ electronics at all times" | -0.5 to -5.0 |
| `budget_cap` | "Daily spending capped at $350" | -0.5 to -1.0 |
| `shipping_rule` | "Groceries must use fast shipping" | -0.5 to -1.0 |
| `price_range` | "Electronics priced at 1.2x or higher" | -0.5 to -1.5 |
| `force_liquidate` | "Liquidate ALL furniture by day 26" | -3.0 to -5.0 |
| `order_freeze` | "No new orders until day 75" | -1.0 to -3.0 |
| `waste_limit` | "Zero grocery waste for days 66-80" | -1.0 to -3.0 |
| `min_cash` | "Maintain cash above $300" | -2.0 |
| `order_limit` | "Order max 4 products per day" | -0.5 to -1.0 |

Key mechanics:
- **Modifications**: Directive H08 says "CORRECTION to H03: spending limit increased to $500". H03 becomes inactive, H08 is the new rule.
- **Expirations**: Some directives have expiry dates and automatically deactivate.
- **Conflicts**: Two active rules with opposing requirements (see above).
- **Deception**: Rules that look beneficial but trigger cascading failures (see above).

## Tasks (Easy / Medium / Hard)

| Task | Cash | Stock | Directives | Events | Key Challenge |
|------|------|-------|-----------|--------|---------------|
| **Easy** | $2,000 | High | 5 | None | Basic compliance + memory |
| **Medium** | $1,500 | Medium | 15 (1 deceptive) | 6 seasonal | Modifications + seasonal planning |
| **Hard** | $1,000 | Low | 27 (2 conflicts, 1 deceptive) | 12 packed | Strategic violation + error recovery |

### Hard Task Highlights
- 27 enforceable directives with modifications, expirations, conflicts, and traps
- 12 events including supply disruptions and competitor launches
- 80% warehouse capacity (reduced)
- Higher base demand with tighter margins
- Requires: genuine 90-day planning, memory management, penalty calculus, and error recovery

## Grading (0.0 - 1.0)

Each task is scored by comparing agent profit against two deterministic baselines:
- **Floor**: Passive agent (never buys, sells initial stock until depleted)
- **Ceiling**: Heuristic agent with perfect demand knowledge and optimal shipping

```
score = clamp((agent_profit - floor) / (ceiling - floor), 0.002, 0.998)
```

Both baselines are deterministic (seeded RNG) and computed fresh each run for reproducibility.

## Step Execution Order

Each `step()` call processes in this order:
1. Save agent memory (notes_to_self, weekly_plan)
2. Weekly reset (spend/waste counters)
3. Issue new directives, expire old ones
4. Tick event countdowns
5. Expire groceries (shelf life = 0)
6. Receive arriving deliveries
7. Process purchase orders (deduct cash, schedule with jitter)
8. Generate demand (base + weekend + events + elasticity)
9. Sell products FIFO
10. Process liquidation FIFO
11. Check directive compliance → violations
12. Check milestones → bonuses
13. Compute decomposed reward
14. Return observation

## Setup

```bash
# Install dependencies
pip install openenv-core[core] fastapi uvicorn pydantic openai python-dotenv

# Run grader baselines
python -c "from server.grader import compute_baselines; [print(f'{t}: floor={f:.2f}, ceiling={c:.2f}') for t in ['easy','medium','hard'] for f,c in [compute_baselines(t)]]"

# Start server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
```

## Running Inference

```bash
# Using HuggingFace Router
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-32B"
export HF_TOKEN="your-token"
python inference.py

# Using OpenAI
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export API_KEY="sk-your-key"
python inference.py
```

## Docker

```bash
docker build -t quartermaster-env .
docker run -p 8000:8000 quartermaster-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns 200 if server is running |
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Submit an action (JSON body), returns next observation with reward |
| `/state` | GET | Get current episode state (day, cash, inventory) |
| `/tasks` | GET | List all 3 tasks with full config |
| `/grader` | POST | Score an episode given task name and agent profit |

### Example Queries

```bash
# List all tasks
curl http://localhost:8000/tasks

# Grade a specific profit
curl -X POST "http://localhost:8000/grader?task_name=easy&agent_profit=5000"

# Run baseline inference (requires API keys in env)
curl "http://localhost:8000/baseline?task_name=easy"
```

## Project Structure

```
├── models.py              # InventoryAction, InventoryObservation, InventoryState (Pydantic)
├── client.py              # EnvClient for remote WebSocket connections
├── inference.py           # LLM inference script (OpenAI-compatible, runs all 3 tasks)
├── generate_sft_data.py   # Generate SFT training data from a strong model
├── generate_grpo_data.py  # Generate GRPO training prompts with replay actions
├── sft_train.py           # Supervised fine-tuning with Unsloth + TRL
├── grpo_train.py          # GRPO reinforcement learning with real env rewards (Dr. GRPO + DAPO tricks)
├── openenv.yaml           # OpenEnv spec manifest
├── pyproject.toml         # Python dependencies
├── Dockerfile             # Container build
├── server/
│   ├── app.py             # FastAPI server (create_app + custom endpoints)
│   ├── inventory_env.py   # Environment (reset/step/state + reward + R_planning)
│   ├── directives.py      # Directive compliance engine (issue/expire/check)
│   ├── constants.py       # All configs: products, events, directives, milestones, tasks
│   └── grader.py          # Floor/ceiling baselines and 0.0-1.0 scoring
```

## GRPO Training Tricks

Our GRPO training pipeline incorporates three research-backed optimizations from recent papers ([DAPO](https://huggingface.co/papers/2503.14476), [Dr. GRPO](https://huggingface.co/papers/2503.20783)) that improve training stability and sample efficiency:

| Trick | Parameter | Value | What It Does |
|-------|-----------|-------|-------------|
| **DAPO Asymmetric Clipping** | `epsilon_low` / `epsilon_high` | 0.2 / 0.28 | Widens the upper clipping bound to prevent entropy collapse. Low-probability tokens (exploration) are less aggressively clipped, preserving the model's ability to discover novel strategies. Standard symmetric clipping (ε=0.2) suppresses exploration too early. |
| **Dr. GRPO Loss** | `loss_type` | `"dr_grpo"` | Uses a fixed denominator for loss normalization instead of per-sequence length. This removes response-length bias: under standard GRPO, longer incorrect completions receive softer gradients than short incorrect ones, creating a perverse incentive. Dr. GRPO ensures equal gradient magnitude regardless of completion length. |
| **Truncation Masking** | `mask_truncated_completions` | `True` | Completions that exceed `max_completion_length` are masked out of the loss entirely, rather than receiving a negative reward. Penalizing truncated completions confuses the model — it can't distinguish "bad reasoning" from "good reasoning that ran out of tokens." Masking avoids this signal corruption. |

### Why These Tricks Matter for Inventory Management

- **Asymmetric clipping** is critical because the agent needs to explore unconventional strategies (e.g., intentionally violating a low-penalty directive to satisfy a high-penalty one). Standard clipping suppresses this exploration.
- **Dr. GRPO loss** matters because `notes_to_self` and `weekly_plan` create variable-length completions. An agent that writes detailed planning notes (longer output) shouldn't get softer penalties when those plans are wrong.
- **Truncation masking** prevents the model from learning to write shorter, less detailed notes just to avoid truncation penalties — which would undermine our content-aware planning reward (`R_planning`).