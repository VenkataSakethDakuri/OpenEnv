---
title: Inventory Optimization Environment
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Retail Inventory Optimization Environment

An OpenEnv reinforcement learning environment that simulates day-by-day retail inventory management across 5 product categories. An AI agent must balance purchasing, pricing, shipping, and liquidation decisions to maximize profit over a 30-day episode.

## Why Inventory Management?

Retail inventory optimization is a real-world task performed daily by store managers, warehouse operators, and supply chain planners. The agent faces the same challenges as a human manager: uncertain demand, perishable goods, shipping delays, seasonal events, and limited cash flow. Poor decisions lead to stockouts (lost sales), waste (expired goods), or cash tied up in unsold inventory.

## Environment Description

You manage a retail store selling 5 products with different characteristics:

| Product | Sell Price | Cost Price | Profit Margin | Shelf Life |
|---------|-----------|------------|---------------|------------|
| Electronics | $150 | $100 | $50 | No expiry |
| Clothing | $40 | $25 | $15 | No expiry |
| Groceries | $10 | $5 | $5 | 5 days |
| Furniture | $200 | $130 | $70 | No expiry |
| Toys | $25 | $12 | $13 | No expiry |

Each day the agent receives the current store state (cash, inventory with batch expiry, pending deliveries, upcoming events) and must decide:
- **What to buy** and how much of each product
- **How to ship** — slow (cheap but unreliable), medium, or fast (expensive but guaranteed)
- **What to liquidate** — dispose of expiring or excess stock
- **How to price** — set per-product price multipliers that affect demand via elasticity

Customer demand is generated each day based on base ranges, weekend boosts (1.2x on days 5-6), and seasonal event multipliers (up to 3x during Black Friday, Christmas, etc.). The agent cannot see future demand — only yesterday's demand as feedback.

The episode runs for 30 days. The goal is to maximize total profit.

## Environment Design Highlights

### Batch-Tracked Inventory with FIFO
Inventory is tracked per batch with individual expiry dates. Groceries expire after 5 days. Selling and liquidation follow FIFO (First In, First Out) — oldest batches are consumed first, mimicking real warehouse operations.

```json
{"groceries": [[20, 3], [15, 5], [10, 1]]}
```
Three batches: 20 units (3 days left), 15 units (5 days left), 10 units (1 day left — liquidate or lose them).

### Dynamic Pricing with Price Elasticity
The agent can set per-product price multipliers (0.5x to 1.5x) each day. Demand responds to pricing via realistic elasticity values — groceries are inelastic (people buy regardless), while clothing and toys are highly elastic (price-sensitive customers).

| Product | Elasticity | Effect of 1.3x price |
|---------|-----------|----------------------|
| Electronics | 1.2 | Demand drops ~24% |
| Clothing | 1.5 | Demand drops ~38% |
| Groceries | 0.4 | Demand drops only ~11% |
| Furniture | 0.8 | Demand drops ~22% |
| Toys | 1.3 | Demand drops ~33% |

### Delivery Jitter
Shipping isn't perfectly reliable. Slow delivery has +/-2 day variance, medium has +/-1 day. Only fast delivery (at 5x the cost) is guaranteed next-day. The agent must account for uncertainty when planning restocks before events.

### Seasonal Events with Demand Spikes
Five events are spread across the 30-day episode. Each event triggers a 2-day demand multiplier — Black Friday triples electronics demand, Christmas triples toys, etc. A "new competitor" event actually reduces demand. The agent sees countdowns and must stock up in advance.

### Decomposed Per-Step Reward
The reward function provides granular feedback every step, not just end-of-episode:

| Signal | Formula | Purpose |
|--------|---------|---------|
| Successful sales | `+sold * sell_price * 0.001` | Reward revenue proportional to product value |
| Missed sales | `-missed * sell_price * 0.001` | Penalize stockouts, weighted by product value |
| Expired groceries | `-0.05 * expired_count` | Penalize waste from overbuying perishables |
| Failed purchases | `-0.5 per rejected order` | Penalize ordering beyond cash budget |
| Liquidation loss | `-disposed_value * 0.001` | Penalize disposal proportional to cost |

### Conversation History for LLM Agents
The inference script maintains a rolling 7-day conversation history. The LLM sees its past observations and decisions, enabling it to spot demand trends, learn from mistakes, and adjust strategy across the episode.

## Action Space

```python
class InventoryAction(Action):
    buy_quantities: Dict[str, int] = {}
    delivery_method: Literal["slow", "medium", "fast"] = "slow"
    liquidate: Dict[str, int] = {}
    price_multipliers: Dict[str, float] = {}
```

| Field | Description |
|-------|-------------|
| `buy_quantities` | Products and amounts to order. Empty `{}` to skip buying. |
| `delivery_method` | `"slow"` ($2/unit, 3-7 days), `"medium"` ($5/unit, 2-4 days), `"fast"` ($10/unit, 1 day guaranteed) |
| `liquidate` | Products and amounts to dispose of (no revenue). Use for expiring groceries or freeing warehouse space. |
| `price_multipliers` | Per-product selling price multiplier (0.5-1.5). Affects demand via elasticity. Default 1.0 if omitted. |

## Observation Space

```python
class InventoryObservation(Observation):
    current_day: int
    total_cash: float
    day_profit: float
    total_profit: float
    demand_today: Dict[str, int]           # yesterday's demand (feedback)
    updated_inventory: Dict[str, List]     # [[qty, days_left], ...] per batch
    remaining_capacity: Dict[str, int]     # warehouse space left per product
    updated_events: Dict[str, int]         # event countdowns (negative = active/ended)
    updated_deliveries: List[Dict]         # in-transit shipments
```

## Tasks (Easy / Medium / Hard)

### Easy — "Steady State"
- Low starting stock, low steady demand, no events
- Starting cash: $1,000 | Full warehouse capacity
- Agent needs to restock regularly but demand is predictable
- No events, no demand spikes — pure supply chain management

### Medium — "Seasonal Rush"
- Default stock/cash, all 5 events spread across 30 days
- Events: Black Friday (day 6), Christmas (day 12), Back to School (day 18), Summer Clearance (day 24), New Competitor (day 28)
- Agent must anticipate demand spikes and restock before events hit

### Hard — "Chaos Mode"
- Half starting cash ($500), low stock, events packed close together (days 4, 8, 12, 16, 20)
- Higher base demand, smaller warehouse capacity
- Agent must balance tight budget, overlapping event spikes, perishable goods, and limited storage

## Grading (0.0 - 1.0)

Each task is scored by comparing agent profit against two deterministic baselines:
- **Floor**: Passive agent that never buys (sells initial stock until depleted)
- **Ceiling**: Theoretical max profit assuming perfect demand knowledge and cheapest shipping

```
score = clamp((agent_profit - floor) / (ceiling - floor), 0.0, 1.0)
```

Both baselines are deterministic (seeded RNG) and computed fresh each run to ensure reproducibility.

## Setup

```bash
# Install dependencies
pip install openenv-core[core] fastapi uvicorn pydantic openai numpy python-dotenv

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
docker build -t inventory-env .
docker run -p 8000:8000 inventory-env
```

## Step Execution Order

Each `step()` call processes in this order:
1. Tick event countdowns (into negatives to track active duration)
2. Remove expired groceries (shelf life = 0)
3. Receive arriving deliveries (add to inventory with fresh shelf life)
4. Process purchase orders (deduct cash, schedule deliveries with jitter)
5. Generate demand (base + weekend boost + event multipliers + price elasticity)
6. Sell products FIFO (oldest batches first, track missed sales)
7. Liquidate requested stock FIFO (no revenue)
8. Compute profit, reward, update state, return observation

## Project Structure

```
├── models.py              # InventoryAction, InventoryObservation, InventoryState (Pydantic)
├── client.py              # EnvClient for remote WebSocket connections
├── inference.py           # LLM inference script with conversation history (runs all 3 tasks)
├── openenv.yaml           # OpenEnv spec manifest
├── pyproject.toml         # Python dependencies
├── Dockerfile             # Multi-stage container build from openenv-base
├── server/
│   ├── app.py             # FastAPI server (create_app + uvicorn entry point)
│   ├── inventory_env.py   # Environment (reset, step, state, demand generation)
│   ├── constants.py       # All configs: prices, stock, events, tasks, elasticity
│   └── grader.py          # Floor/ceiling baselines and 0.0-1.0 scoring
└── scripts/
    └── validate-submission.sh  # Pre-submission validator
```