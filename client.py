from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import InventoryAction, InventoryObservation, InventoryState


class InventoryEnv(EnvClient[InventoryAction, InventoryObservation, InventoryState]):

    def _step_payload(self, action: InventoryAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}

        if action.buy_quantities:
            payload["buy_quantities"] = action.buy_quantities
        if action.delivery_methods:
            payload["delivery_methods"] = action.delivery_methods
        if action.liquidate:
            payload["liquidate"] = action.liquidate
        if action.price_multipliers:
            payload["price_multipliers"] = action.price_multipliers
        if action.notes_to_self:
            payload["notes_to_self"] = action.notes_to_self
        if action.weekly_plan is not None:
            payload["weekly_plan"] = action.weekly_plan

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[InventoryObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False)

        observation = InventoryObservation(
            current_day=obs_data.get("current_day", 0),
            total_days=obs_data.get("total_days", 90),
            total_cash=obs_data.get("total_cash", 0),
            day_profit=obs_data.get("day_profit", 0),
            total_profit=obs_data.get("total_profit", 0),
            demand_today=obs_data.get("demand_today", {}),
            updated_inventory=obs_data.get("updated_inventory", {}),
            remaining_capacity=obs_data.get("remaining_capacity", {}),
            updated_events=obs_data.get("updated_events", {}),
            updated_deliveries=obs_data.get("updated_deliveries", []),
            new_directives=obs_data.get("new_directives", []),
            active_directive_ids=obs_data.get("active_directive_ids", []),
            directive_violations_last_step=obs_data.get("directive_violations_last_step", []),
            milestones=obs_data.get("milestones", {}),
            agent_notes=obs_data.get("agent_notes", ""),
            agent_weekly_plan=obs_data.get("agent_weekly_plan", ""),
            done=done,
            reward=reward,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> InventoryState:
        return InventoryState(
            episode_id=payload.get("episode_id", ""),
            current_day=payload.get("current_day", 0),
            total_days=payload.get("total_days", 90),
            cash=payload.get("cash", 0.0),
            total_profit=payload.get("total_profit", 0.0),
            inventory=payload.get("inventory", {}),
            active_directives=payload.get("active_directives", 0),
            total_violations=payload.get("total_violations", 0),
            milestones_achieved=payload.get("milestones_achieved", 0),
            milestones_total=payload.get("milestones_total", 0),
        )