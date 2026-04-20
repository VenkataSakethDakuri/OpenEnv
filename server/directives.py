"""
Directive compliance engine for long-horizon inventory environment.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ActiveDirective:
    id: str
    day_issued: int
    type: str
    text: str
    params: Dict[str, Any]
    penalty: float
    expires: Optional[int]
    modifies: Optional[str]
    active: bool = True


class DirectiveEngine:
    def __init__(self, directives_config: List[Dict]):
        self.all_directives = directives_config
        self.active: Dict[str, ActiveDirective] = {}
        self.total_violations = 0

    def advance_day(self, current_day: int) -> List[ActiveDirective]:
        """Issue new directives, expire old ones. Returns newly issued."""
        new_directives = []

        for d in self.all_directives:
            if d["day"] == current_day:
                directive = ActiveDirective(
                    id=d["id"], day_issued=current_day, type=d["type"],
                    text=d["text"], params=d.get("params", {}),
                    penalty=d.get("penalty", 0.0), expires=d.get("expires"),
                    modifies=d.get("modifies"),
                )
                if directive.modifies and directive.modifies in self.active:
                    self.active[directive.modifies].active = False
                self.active[directive.id] = directive
                new_directives.append(directive)

        for d in self.active.values():
            if d.active and d.expires is not None and current_day > d.expires:
                d.active = False

        return new_directives

    def check_compliance(self, current_day: int, env_state: Dict, action_data: Dict) -> List[Dict]:
        """Check active directives. Returns list of violations."""
        violations = []

        for directive in self.active.values():
            if not directive.active:
                continue
            if self._check_violated(directive, current_day, env_state, action_data):
                violations.append({
                    "id": directive.id,
                    "text": directive.text[:80],
                    "penalty": directive.penalty,
                })
                self.total_violations += 1

        return violations

    def _check_violated(self, d: ActiveDirective, day: int, state: Dict, action: Dict) -> bool:
        p = d.params

        if d.type == "min_stock":
            product = p["product"]
            min_qty = p["min_qty"]
            if product == "all":
                for prod, batches in state["inventory"].items():
                    if sum(b[0] for b in batches) < min_qty:
                        return True
                return False
            batches = state["inventory"].get(product, [])
            return sum(b[0] for b in batches) < min_qty

        elif d.type == "shipping_rule":
            buy_qty = action.get("buy_quantities", {}).get(p["product"], 0)
            if buy_qty > 0:
                method = action.get("delivery_methods", {}).get(p["product"], "slow")
                return method not in p["allowed_methods"]
            return False

        elif d.type == "budget_cap":
            if p["period"] == "daily":
                return state["daily_spend"] > p["max_amount"]
            return state["weekly_spend"] > p["max_amount"]

        elif d.type == "profit_target":
            if day == p["deadline"]:
                return state["total_profit"] < p["target"]
            return False

        elif d.type == "price_range":
            mult = action.get("price_multipliers", {}).get(p["product"], 1.0)
            return mult < p["min_mult"] or mult > p["max_mult"]

        elif d.type == "order_limit":
            ordered = sum(1 for v in action.get("buy_quantities", {}).values() if v > 0)
            if "min_products" in p and ordered > 0:
                if ordered < p["min_products"]:
                    return True
            return ordered > p["max_products"]

        elif d.type == "waste_limit":
            return state.get("weekly_waste", 0) > p["max_units"]

        elif d.type == "min_cash":
            return state["cash"] < p["min_amount"]

        elif d.type == "force_liquidate":
            if day == p["deadline"]:
                batches = state["inventory"].get(p["product"], [])
                return sum(b[0] for b in batches) > 0
            return False

        elif d.type == "target_stock":
            if day == p["deadline"]:
                batches = state["inventory"].get(p["product"], [])
                return sum(b[0] for b in batches) < p["min_qty"]
            return False

        elif d.type == "order_freeze":
            if day > p["after_day"]:
                return any(v > 0 for v in action.get("buy_quantities", {}).values())
            return False

        return False

    def get_active_ids(self) -> List[str]:
        return [d.id for d in self.active.values() if d.active]

    def get_active_count(self) -> int:
        return sum(1 for d in self.active.values() if d.active)