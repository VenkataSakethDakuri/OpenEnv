from openenv.core.env_server.interfaces import Environment
import copy
import random
from uuid import uuid4

from models import InventoryAction, InventoryObservation, InventoryState
from .constants import (
    BASE_PRICES, COST_PRICES, SHELF_LIFE, SHIPPING_COST, SHIPPING_DAYS,
    EXTRA_INVENTORY_COST, WEEKEND_MULTIPLIER,
    EVENT_EFFECTS, EVENT_DURATION, PRICE_ELASTICITY, TASKS,
    LOAN_AMOUNT, LOAN_DAILY_INTEREST, LOAN_REVENUE_REPAYMENT,
    LOAN_ELIGIBILITY_THRESHOLD, MAX_LOANS,
)
from .directives import DirectiveEngine


def _build_inventory(stock):
    inv = {}
    for product, qty in stock.items():
        shelf = SHELF_LIFE[product]
        inv[product] = [[qty, shelf]]
    return inv


class InventoryEnvironment(Environment):

    def __init__(self, task_name="medium"):
        super().__init__()
        self.task_name = task_name
        self.task = TASKS[task_name]
        self.reset()

    def reset(self, seed=None, episode_id=None, **kwargs) -> InventoryObservation:
        self.seed = seed if seed is not None else self.task["seed"]
        self.cash = self.task["initial_cash"]
        self.inventory = _build_inventory(self.task["initial_stock"])
        self.events = copy.deepcopy(self.task["events"])
        self.deliveries = []
        self.current_day = 0
        self.total_profit = 0.0
        self.reward = 0.0
        self.max_days = self.task["max_days"]
        self.inventory_capacity = copy.deepcopy(self.task["inventory_capacity"])
        self.base_demand = self.task["base_demand"]
        self.consecutive_idle_days = 0

        # Long-horizon state
        self.directive_engine = DirectiveEngine(self.task["directives"])
        self.milestones_achieved = set()
        self.agent_notes = ""
        self.agent_weekly_plan = ""
        self.weekly_spend = 0.0
        self.weekly_waste = 0
        self.week_start_day = 1
        self.total_violations = 0
        self.total_waste = 0
        self.grocery_waste_streak = 0
        self._prev_notes = ""

        # Loan state
        self.loan_balance = 0.0
        self.loans_taken = 0

        self._state = InventoryState(
            episode_id=str(uuid4()),
            current_day=0,
            total_days=self.max_days,
            cash=self.cash,
            total_profit=0.0,
            inventory={p: sum(b[0] for b in self.inventory[p]) for p in self.inventory},
            active_directives=0,
            total_violations=0,
            milestones_achieved=0,
            milestones_total=len(self.task["milestones"]),
            loan_balance=0.0,
            loans_taken=0,
        )

        return InventoryObservation(
            current_day=0,
            total_days=self.max_days,
            total_cash=self.cash,
            day_profit=0.0,
            total_profit=0.0,
            demand_today={},
            updated_inventory=copy.deepcopy(self.inventory),
            remaining_capacity={p: max(0, self.inventory_capacity[p] - sum(b[0] for b in self.inventory[p])) for p in self.inventory},
            updated_events=copy.deepcopy(self.events),
            updated_deliveries=[],
            new_directives=[],
            active_directive_ids=[],
            directive_violations_last_step=[],
            milestones=self._milestone_status(),
            agent_notes="",
            agent_weekly_plan="",
            loan_balance=0.0,
            loans_taken=0,
            loans_remaining=MAX_LOANS,
            reward=0.0,
            done=False,
        )

    def step(self, action: InventoryAction, timeout_s=None, **kwargs) -> InventoryObservation:
        self.current_day += 1
        self.reward = 0.0
        day_cost = 0.0
        day_revenue = 0.0

        # Save agent memory
        if action.notes_to_self:
            self.agent_notes = action.notes_to_self
        if action.weekly_plan is not None:
            self.agent_weekly_plan = action.weekly_plan

        # --- Loan processing ---
        if action.take_loan and self.cash < LOAN_ELIGIBILITY_THRESHOLD and self.loans_taken < MAX_LOANS:
            self.cash += LOAN_AMOUNT
            self.loan_balance += LOAN_AMOUNT
            self.loans_taken += 1

        # Compound interest on outstanding loan
        if self.loan_balance > 0:
            self.loan_balance *= (1.0 + LOAN_DAILY_INTEREST)

        # Weekly reset
        if (self.current_day - self.week_start_day) >= 7:
            self.weekly_spend = 0.0
            self.weekly_waste = 0
            self.week_start_day = self.current_day

        # Issue directives
        new_directives = self.directive_engine.advance_day(self.current_day)

        # Tick events
        for event_name in self.events:
            self.events[event_name] -= 1

        total_inventory = sum(sum(b[0] for b in self.inventory[p]) for p in self.inventory)

        # Expire groceries
        expired_count = 0
        new_batches = []
        for batch in self.inventory["groceries"]:
            if batch[1] == 0:
                expired_count += batch[0]
            else:
                new_batches.append([batch[0], batch[1] - 1])
        self.inventory["groceries"] = new_batches
        self.total_waste += expired_count
        self.weekly_waste += expired_count
        if expired_count > 0:
            self.grocery_waste_streak = 0
        else:
            self.grocery_waste_streak += 1

        # Handle deliveries
        remaining_deliveries = []
        total_delivered = 0
        for delivery in self.deliveries:
            for product, shipment in delivery.items():
                qty, arrival_day = shipment
                if arrival_day <= self.current_day:
                    total_delivered += qty
                    self.inventory[product].append([qty, SHELF_LIFE[product]])
                else:
                    remaining_deliveries.append(delivery)
        self.deliveries = remaining_deliveries

        # Process purchases
        had_unaffordable = False
        for product, qty in action.buy_quantities.items():
            if qty <= 0 or product not in BASE_PRICES:
                continue
            method = action.delivery_methods.get(product, "slow")
            unit_cost = COST_PRICES[product] + SHIPPING_COST[method]
            current_qty = sum(b[0] for b in self.inventory[product])
            overage = max(0, (current_qty + qty) - self.inventory_capacity[product])
            extra_cost = overage * EXTRA_INVENTORY_COST[product]
            total_cost = qty * unit_cost + extra_cost

            if total_cost > self.cash:
                had_unaffordable = True
                continue

            self.cash -= total_cost
            day_cost += total_cost

            arrival_day = self.current_day + SHIPPING_DAYS[method]
            jitter_rng = random.Random(self.seed * 2000 + self.current_day * 100 + hash(product))
            if method == "slow":
                arrival_day += jitter_rng.randint(-2, 2)
            elif method == "medium":
                arrival_day += jitter_rng.randint(-1, 1)
            arrival_day = max(self.current_day + 1, arrival_day)
            self.deliveries.append({product: [qty, arrival_day]})

        self.weekly_spend += day_cost

        # Generate demand
        demand = self._generate_demand()

        # Price elasticity
        price_mults = {}
        for product in demand:
            pm = max(0.5, min(1.5, action.price_multipliers.get(product, 1.0)))
            price_mults[product] = pm
            e = PRICE_ELASTICITY[product]
            demand[product] = max(0, int(demand[product] * pm ** -e))

        # Sell (FIFO)
        max_daily_revenue = 0.0
        total_demand_units = 0
        total_sold = 0
        for product, demand_qty in demand.items():
            sell_price = BASE_PRICES[product] * price_mults[product]
            max_daily_revenue += demand_qty * sell_price
            total_demand_units += demand_qty
            available = sum(b[0] for b in self.inventory[product])

            if demand_qty > available:
                sold = available
                self.inventory[product] = []
            else:
                sold = demand_qty
                remaining = demand_qty
                new_batches = []
                for batch in self.inventory[product]:
                    if remaining <= 0:
                        new_batches.append(batch)
                    elif batch[0] <= remaining:
                        remaining -= batch[0]
                    else:
                        new_batches.append([batch[0] - remaining, batch[1]])
                        remaining = 0
                self.inventory[product] = new_batches

            total_sold += sold
            day_revenue += sold * sell_price

        # Liquidate
        liquidated_units = 0
        for product, count in action.liquidate.items():
            if product not in self.inventory or count <= 0:
                continue
            available = sum(b[0] for b in self.inventory[product])
            actually_removed = min(count, available)
            liquidated_units += actually_removed
            remaining = count
            new_batches = []
            for batch in self.inventory[product]:
                if remaining <= 0:
                    new_batches.append(batch)
                elif batch[0] <= remaining:
                    remaining -= batch[0]
                else:
                    new_batches.append([batch[0] - remaining, batch[1]])
                    remaining = 0
            self.inventory[product] = new_batches

        self.total_waste += liquidated_units
        self.weekly_waste += liquidated_units

        # Financials
        day_profit = day_revenue - day_cost
        self.cash += day_revenue
        self.total_profit += day_profit
        done = self.current_day >= self.max_days

        # Loan repayment: 15% of daily revenue auto-deducted
        loan_repayment = 0.0
        if self.loan_balance > 0 and day_revenue > 0:
            loan_repayment = min(day_revenue * LOAN_REVENUE_REPAYMENT, self.loan_balance)
            self.cash -= loan_repayment
            self.loan_balance -= loan_repayment

        # End of episode: subtract remaining loan balance from total profit and cash
        if done and self.loan_balance > 0:
            self.total_profit -= self.loan_balance
            self.cash -= self.loan_balance
            self.loan_balance = 0.0

        # --- Directive compliance ---
        env_state = {
            "inventory": self.inventory,
            "cash": self.cash,
            "total_profit": self.total_profit,
            "daily_spend": day_cost,
            "weekly_spend": self.weekly_spend,
            "weekly_waste": self.weekly_waste,
        }
        action_data = {
            "buy_quantities": action.buy_quantities,
            "delivery_methods": action.delivery_methods,
            "liquidate": action.liquidate,
            "price_multipliers": action.price_multipliers,
        }
        violations = self.directive_engine.check_compliance(self.current_day, env_state, action_data)
        self.total_violations += len(violations)

        # --- Milestones ---
        milestone_bonus = self._check_milestones()

        # --- Reward ---
        # Dense signals (sum to 1.0 for base, range [-1.0, +1.0])
        R_revenue = 2.0 * (day_revenue / max(max_daily_revenue, 1.0)) - 1.0
        R_fulfillment = 2.0 * (total_sold / max(total_demand_units, 1)) - 1.0

        total_managed = total_inventory + total_delivered
        waste_rate = (expired_count + liquidated_units) / max(total_managed, 1)
        R_waste = max(-1.0, 1.0 - 2.0 * min(waste_rate * 3, 1.0))

        active_checkable = sum(1 for d in self.directive_engine.active.values()
                               if d.active)
        if active_checkable > 0:
            R_directives = 1.0 - (2.0 * len(violations) / active_checkable)
        else:
            R_directives = 1.0
        R_directives = max(-1.0, min(1.0, R_directives))

        R_planning = self._compute_R_planning(action, violations)

        # Hard fails
        hard_penalty = 0.0
        if had_unaffordable:
            hard_penalty -= 1.0
        if self.cash < 10 and self.loans_taken >= MAX_LOANS:
            hard_penalty -= 2.0  # true bankruptcy: no more loans available

        is_idle = (not action.buy_quantities or all(v == 0 for v in action.buy_quantities.values())) and \
                  (not action.liquidate or all(v == 0 for v in action.liquidate.values()))
        if is_idle:
            self.consecutive_idle_days += 1
        else:
            self.consecutive_idle_days = 0
        if self.consecutive_idle_days >= 3:
            hard_penalty -= 1.0

        # Directive violation penalties
        directive_penalty = sum(v["penalty"] for v in violations)

        # Dense per-step signals (weighted sum = 1.0, range [-1, +1])
        dense_reward = (
            0.40 * R_directives +
            0.20 * R_planning +
            0.15 * R_revenue +
            0.15 * R_fulfillment +
            0.10 * R_waste
        )

        # Sparse signals (event-driven, not every step)
        sparse_reward = (
            milestone_bonus +        # +1.5 to +5.0 on milestone achievement
            directive_penalty +      # -0.3 to -5.0 per violation
            hard_penalty             # -1.0 to -2.0 for hard fails
        )

        self.reward = dense_reward + sparse_reward

        # Expose sub-components for training diagnostics
        self.reward_components = {
            "R_directives": R_directives,
            "R_planning": R_planning,
            "R_revenue": R_revenue,
            "R_fulfillment": R_fulfillment,
            "R_waste": R_waste,
            "milestone_bonus": milestone_bonus,
            "directive_penalty": directive_penalty,
            "hard_penalty": hard_penalty,
        }

        # Update state
        self._state = InventoryState(
            episode_id=self._state.episode_id,
            current_day=self.current_day,
            total_days=self.max_days,
            cash=self.cash,
            total_profit=self.total_profit,
            inventory={p: sum(b[0] for b in self.inventory[p]) for p in self.inventory},
            active_directives=self.directive_engine.get_active_count(),
            total_violations=self.total_violations,
            milestones_achieved=len(self.milestones_achieved),
            milestones_total=len(self.task["milestones"]),
            loan_balance=round(self.loan_balance, 2),
            loans_taken=self.loans_taken,
        )

        return InventoryObservation(
            current_day=self.current_day,
            total_days=self.max_days,
            total_cash=self.cash,
            day_profit=day_profit,
            total_profit=self.total_profit,
            demand_today=demand,
            updated_inventory=copy.deepcopy(self.inventory),
            remaining_capacity={p: max(0, self.inventory_capacity[p] - sum(b[0] for b in self.inventory[p])) for p in self.inventory},
            updated_events=copy.deepcopy(self.events),
            updated_deliveries=copy.deepcopy(self.deliveries),
            new_directives=[{
                "id": d.id, "type": d.type, "text": d.text,
                "expires_day": d.expires, "replaces": d.modifies,
            } for d in new_directives],
            active_directive_ids=self.directive_engine.get_active_ids(),
            directive_violations_last_step=violations,
            milestones=self._milestone_status(),
            agent_notes=self.agent_notes,
            agent_weekly_plan=self.agent_weekly_plan,
            loan_balance=round(self.loan_balance, 2),
            loans_taken=self.loans_taken,
            loans_remaining=MAX_LOANS - self.loans_taken,
            reward=self.reward,
            done=done,
        )

    def _compute_R_planning(self, action, violations):
        """Content-aware planning reward. Range: [-1.0, +1.0]."""
        notes = action.notes_to_self or ""
        plan = action.weekly_plan or ""

        # No notes and no plan = worst case
        if not notes and not plan:
            self._prev_notes = ""
            return -1.0

        score = -0.5  # Start negative; must earn your way to positive

        # 1. Directive tracking (up to +0.50)
        active_ids = self.directive_engine.get_active_ids()
        if active_ids:
            ids_mentioned = sum(1 for d_id in active_ids if d_id in notes)
            score += 0.50 * (ids_mentioned / len(active_ids))

        # 2. Situational awareness (up to +0.30)
        products = ["electronics", "clothing", "groceries", "furniture", "toys"]
        products_mentioned = sum(1 for p in products if p in notes.lower() or p in plan.lower())
        has_numbers = sum(1 for c in notes if c.isdigit()) > 3
        score += (min(products_mentioned, 3) / 3) * 0.15
        score += 0.15 if has_numbers else 0.0

        # 3. Note evolution (up to +0.30, penalty for copy-paste)
        if self._prev_notes:
            if notes == self._prev_notes:
                score -= 0.30  # copy-paste = big penalty
            elif len(notes) > 30:
                score += 0.30  # evolved notes = full credit
        else:
            score += 0.15 if len(notes) > 30 else 0.0
        self._prev_notes = notes

        # 4. Violation acknowledgment (up to +0.20)
        if violations:
            violation_ids = [v['id'] for v in violations]
            acknowledged = sum(1 for v_id in violation_ids if v_id in notes)
            score += 0.20 * (acknowledged / len(violation_ids))

        # 5. Plan structure (up to +0.20)
        if plan:
            plan_words = len(plan.split())
            has_structure = any(m in plan for m in [':', '-', '1.', '2.', '*'])
            score += 0.10 if plan_words > 15 else 0.0
            score += 0.10 if has_structure else 0.0

        return max(-1.0, min(1.0, score))

    def _generate_demand(self):
        rng = random.Random(self.seed * 1000 + self.current_day)
        demand = {}
        for product, (lo, hi) in self.base_demand.items():
            demand[product] = rng.randint(lo, hi)

        if self.current_day % 7 in (5, 6):
            for product in demand:
                demand[product] = int(demand[product] * WEEKEND_MULTIPLIER)

        for event_name, days in self.events.items():
            if -EVENT_DURATION < days <= 0 and event_name in EVENT_EFFECTS:
                for product, mult in EVENT_EFFECTS[event_name].items():
                    demand[product] = int(demand[product] * mult)

        return demand

    def _milestone_status(self):
        status = {}
        for name, m in self.task["milestones"].items():
            current = self._get_milestone_value(m["metric"])
            status[name] = {
                "target": m["target"],
                "deadline": m["deadline"],
                "achieved": name in self.milestones_achieved,
                "current": current,
            }
        return status

    def _get_milestone_value(self, metric):
        if metric == "total_profit":
            return self.total_profit
        elif metric == "waste_rate_below":
            total_through = self.total_waste + sum(sum(b[0] for b in self.inventory[p]) for p in self.inventory)
            return self.total_waste / max(total_through, 1)
        elif metric == "furniture_stock_zero":
            return sum(b[0] for b in self.inventory.get("furniture", []))
        elif metric == "clothing_stock_zero":
            return sum(b[0] for b in self.inventory.get("clothing", []))
        elif metric == "toys_stock_above":
            return sum(b[0] for b in self.inventory.get("toys", []))
        elif metric == "grocery_waste_zero_streak":
            return self.grocery_waste_streak
        return 0.0

    def _check_milestones(self) -> float:
        bonus = 0.0
        for name, m in self.task["milestones"].items():
            if name in self.milestones_achieved:
                continue
            if self.current_day > m["deadline"]:
                continue

            achieved = False
            metric = m["metric"]
            target = m["target"]

            if metric == "total_profit":
                achieved = self.total_profit >= target
            elif metric == "waste_rate_below":
                total_through = self.total_waste + sum(sum(b[0] for b in self.inventory[p]) for p in self.inventory)
                rate = self.total_waste / max(total_through, 1)
                achieved = rate < target
            elif metric == "furniture_stock_zero":
                achieved = sum(b[0] for b in self.inventory.get("furniture", [])) == 0
            elif metric == "clothing_stock_zero":
                achieved = sum(b[0] for b in self.inventory.get("clothing", [])) == 0
            elif metric == "toys_stock_above":
                achieved = sum(b[0] for b in self.inventory.get("toys", [])) >= target
            elif metric == "grocery_waste_zero_streak":
                achieved = self.grocery_waste_streak >= target

            if achieved:
                self.milestones_achieved.add(name)
                bonus += m["bonus"]

        return bonus

    @property
    def state(self) -> InventoryState:
        return self._state