from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import InventoryAction, InventoryObservation, InventoryState



class InventoryEnv(EnvClient[InventoryAction, InventoryObservation, InventoryState]):
    

    def _step_payload(self, action : InventoryAction) -> Dict[str, Any]:

        payload: Dict[str, Any] = {}

        if action.buy_quantities is not None:
            payload["buy_quantities"] = action.buy_quantities

        if action.delivery_method is not None:
            payload["delivery_method"] = action.delivery_method

        if action.liquidate is not None:
            payload["liquidate"] = action.liquidate

        if action.price_multipliers is not None:
            payload["price_multipliers"] = action.price_multipliers

        return payload


    def _parse_result(self, payload: Dict) -> StepResult[InventoryObservation]:

        obs_data = payload.get("observation", {})

        observation = InventoryObservation(

            current_day = obs_data.get("current_day", 0),
            total_cash = obs_data.get("total_cash", 0),
            day_profit = obs_data.get("day_profit", 0),
            total_profit = obs_data.get("total_profit", 0),
            demand_today = obs_data.get("demand_today", {}),
            updated_inventory = obs_data.get("updated_inventory", {}),
            remaining_capacity = obs_data.get("remaining_capacity", {}),
            updated_events = obs_data.get("updated_events", {}),
            updated_deliveries = obs_data.get("updated_deliveries", []),
            done = obs_data.get("done", False),
            reward = obs_data.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation = observation,
            reward = observation.reward,
            done = observation.done,
        )


    def _parse_state(self, payload: Dict[str, Any]) -> InventoryState:

        return InventoryState(
            episode_id = payload.get("episode_id", ""),
            current_day = payload.get("current_day", 0),
            cash = payload.get("cash", 0.0),
            inventory = payload.get("inventory", {}),
        )