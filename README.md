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

An OpenEnv reinforcement learning environment that simulates day-by-day retail inventory management across 5 product categories. An AI agent must decide what to buy, how to ship, and what to liquidate to maximize profit over a 30-day episode.

## Environment Description

You manage a retail store selling 5 products with different characteristics:

| Product | Sell Price | Cost Price | Profit Margin | Shelf Life |
|---------|-----------|------------|---------------|------------|
| Electronics | $150 | $100 | $50 | No expiry |
| Clothing | $40 | $25 | $15 | No expiry |
| Groceries | $10 | $5 | $5 | 5 days |
| Furniture | $200 | $130 | $70 | No expiry |
| Toys | $25 | $12 | $13 | No expiry |

Each day, customer demand is generated (with weekend boosts and event spikes). The agent must keep stock levels high enough to meet demand while managing cash flow, shipping delays, warehouse capacity, and perishable goods.

## Action Space

```python
class InventoryAction(Action):
    buy_quantities: Dict[str, int] = {}        # product -> quantity to order
    delivery_method: Literal["slow", "medium", "fast"] = "slow"
    liquidate: Dict[str, int] = {}             # product -> quantity to dispose
```

| Field | Description |
|-------|-------------|
| `buy_quantities` | Products and amounts to order. Empty `{}` to skip buying. |
| `delivery_method` | `"slow"` ($2/unit, 5 days), `"medium"` ($5/unit, 3 days), `"fast"` ($10/unit, 1 day) |
| `liquidate` | Products and amounts to dispose of (no revenue). Use for expiring groceries or freeing warehouse space. |

## Observation Space

```python
class InventoryObservation(Observation):
    current_day: int
    total_cash: float
    day_profit: float
    total_profit: float
    demand_today: Dict[str, int]
    updated_inventory: Dict[str, List[List[Optional[int]]]]  # [[qty, days_left], ...]
    remaining_capacity: Dict[str, int]
    updated_events: Dict[str, int]
    updated_deliveries: List[Dict[str, List[int]]]
```

The inventory uses a batch format with FIFO selling: `{"groceries": [[20, 3], [10, 5]]}` means 20 units expiring in 3 days and 10 units expiring in 5 days.

## Tasks (Easy / Medium / Hard)

### Easy — "Steady State"
- Low starting stock, low steady demand, no events
- Starting cash: $1,000 | Full warehouse capacity
- Agent needs to restock regularly but demand is predictable

### Medium — "Seasonal Rush"
- Default stock/cash, all 5 events spread across 30 days
- Events: Black Friday (day 6), Christmas (day 12), Back to School (day 18), Summer Clearance (day 24), New Competitor (day 28)
- Agent must anticipate demand spikes and restock accordingly

### Hard — "Chaos Mode"
- Half starting cash ($500), low stock, events packed close together
- Higher demand, smaller warehouse capacity
- Agent must balance tight budget, overlapping event spikes, and fast-expiring groceries

## Reward Function

Per-step reward based on multiple signals:
- **Successful sales**: `+sold_units * sell_price * 0.001` (proportional to revenue)
- **Missed sales**: `-missed_units * sell_price * 0.001` (proportional to lost revenue)
- **Expired groceries**: `-0.05 * expired_count`
- **Failed purchases**: `-0.5` per order that exceeds available cash
- **Liquidation loss**: `-liquidated_value * 0.001` (proportional to cost of disposed stock)

## Grading (0.0 - 1.0)

Each task is scored by comparing agent profit against two baselines:
- **Floor**: Passive agent that never buys (sells initial stock until empty)
- **Ceiling**: Smart heuristic that restocks based on demand and events

```
score = clamp((agent_profit - floor) / (ceiling - floor), 0.0, 1.0)
```

## Setup

```bash
# Install dependencies
pip install openenv-core[core] fastapi uvicorn pydantic openai numpy

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
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

## Docker

```bash
docker build -t inventory-env .
docker run -p 8000:8000 inventory-env
```

## Project Structure

```
V2/
├── models.py              # InventoryAction, InventoryObservation, InventoryState
├── client.py              # EnvClient for remote WebSocket connections
├── inference.py           # LLM inference script (runs all 3 tasks)
├── openenv.yaml           # OpenEnv spec manifest
├── pyproject.toml         # Python dependencies
├── Dockerfile             # Container build
├── server/
│   ├── app.py             # FastAPI server (create_app)
│   ├── inventory_env.py   # Environment (reset, step, state)
│   ├── constants.py       # Prices, stock, events, task configs
│   └── grader.py          # Floor/ceiling baselines and scoring
└── scripts/
    └── validate-submission.sh  # Pre-submission validator
```