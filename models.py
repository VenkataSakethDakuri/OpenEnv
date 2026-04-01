from __future__ import annotations

import json
from openenv.core.env_server import Action, Observation, State
from typing import Literal, Dict, List, Optional
from pydantic import field_validator


class InventoryAction(Action):
    buy_quantities : Dict[str, int] = {}
    delivery_method : Literal["slow", "medium", "fast"] = "slow"
    liquidate : Dict[str, int] = {}
    price_multipliers : Dict[str, float] = {}  # product -> 0.5 to 1.5 (default 1.0)

    @field_validator("buy_quantities", "liquidate", "price_multipliers", mode="before")
    @classmethod
    def parse_dict_strings(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class InventoryObservation(Observation):
    current_day : int
    total_cash : float
    day_profit : float
    total_profit : float
    demand_today : Dict[str, int]  # product -> units demanded today
    updated_inventory : Dict[str, List[List[Optional[int]]]]  # product -> [[qty, days_left], ...] per batch
    remaining_capacity : Dict[str, int]  # product -> remaining warehouse space
    updated_events : Dict[str, int]
    updated_deliveries : List[Dict[str, List[int]]] # product name, (quantity of product, days to arrival)


class InventoryState(State):
    episode_id : str
    current_day : int
    cash : float
    inventory : Dict[str, int]