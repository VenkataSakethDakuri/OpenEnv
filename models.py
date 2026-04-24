from __future__ import annotations

import json
from openenv.core.env_server import Action, Observation, State
from typing import Literal, Dict, List, Optional
from pydantic import field_validator


class InventoryAction(Action):
    buy_quantities: Dict[str, int] = {}
    delivery_methods: Dict[str, Literal["slow", "medium", "fast"]] = {}
    liquidate: Dict[str, int] = {}
    price_multipliers: Dict[str, float] = {}
    notes_to_self: str = ""
    weekly_plan: Optional[str] = None
    take_loan: bool = False

    @field_validator("buy_quantities", "liquidate", "price_multipliers", "delivery_methods", mode="before")
    @classmethod
    def parse_dict_strings(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return {}
        return v if v is not None else {}

    @field_validator("notes_to_self", mode="before")
    @classmethod
    def parse_notes(cls, v):
        return str(v) if v else ""


class InventoryObservation(Observation):
    current_day: int
    total_days: int
    total_cash: float
    day_profit: float
    total_profit: float
    demand_today: Dict[str, int]
    updated_inventory: Dict[str, List[List[Optional[int]]]]
    remaining_capacity: Dict[str, int]
    updated_events: Dict[str, int]
    updated_deliveries: List[Dict[str, List[int]]]
    new_directives: List[Dict] = []
    active_directive_ids: List[str] = []
    directive_violations_last_step: List[Dict] = []
    milestones: Dict[str, Dict] = {}
    agent_notes: str = ""
    agent_weekly_plan: str = ""
    loan_balance: float = 0.0
    loans_taken: int = 0
    loans_remaining: int = 2


class InventoryState(State):
    episode_id: str
    current_day: int
    total_days: int = 90
    cash: float
    total_profit: float = 0.0
    inventory: Dict[str, int] = {}
    active_directives: int = 0
    total_violations: int = 0
    milestones_achieved: int = 0
    milestones_total: int = 0
    loan_balance: float = 0.0
    loans_taken: int = 0