from openenv.core.env_server.interfaces import Environment
import copy
import random
from uuid import uuid4

from models import InventoryAction, InventoryObservation, InventoryState
from .constants import (
    INITIAL_CASH, BASE_PRICES, COST_PRICES, SHELF_LIFE, INITIAL_STOCK,
    EVENTS, SHIPPING_COST, SHIPPING_DAYS, INVENTORY_CAPACITY,
    EXTRA_INVENTORY_COST, BASE_DEMAND, WEEKEND_MULTIPLIER, EVENT_EFFECTS,
    EVENT_DURATION, MAX_DAYS, UPGRADE_DELIVERY_COST, TASKS, PRICE_ELASTICITY
)


def _build_inventory(stock):
    """Convert stock dict to batch format: {product: [[qty, days_left], ...]}"""
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
        self.cash = self.task["initial_cash"]
        self.inventory = _build_inventory(self.task["initial_stock"])
        self.events = copy.deepcopy(self.task["events"])
        self.deliveries = []
        self.current_day = 0
        self.total_profit = 0.0
        self.seed = self.task["seed"]
        self.reward = 0.0
        self.max_days = self.task["max_days"]
        self.inventory_capacity = self.task["inventory_capacity"]
        self.base_demand = self.task["base_demand"]
        self.consecutive_idle_days = 0
        self.reset()

    def reset(self, seed: int = None) -> InventoryObservation:
        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.task["seed"]
        self.cash = self.task["initial_cash"]
        self.inventory = _build_inventory(self.task["initial_stock"])
        self.events = copy.deepcopy(self.task["events"])
        self.deliveries = []
        self.current_day = 0
        self.total_profit = 0.0
        self.reward = 0.0
        self.consecutive_idle_days = 0

        self._state = InventoryState(
            episode_id = str(uuid4()),
            current_day = 0,
            cash = self.task["initial_cash"],
            inventory = dict(self.task["initial_stock"])
        )

        return InventoryObservation(
            current_day = 0,
            total_cash = self.cash,
            day_profit = 0.0,
            total_profit = 0.0,
            demand_today = {},
            updated_inventory = copy.deepcopy(self.inventory),
            remaining_capacity = {p: max(0, self.inventory_capacity[p] - sum(b[0] for b in self.inventory[p])) for p in self.inventory},
            updated_events = copy.deepcopy(self.events),
            updated_deliveries = [],
            reward = 0.0,
            done = False,
        )

    def step(self, action: InventoryAction) -> InventoryObservation:
        self.current_day += 1
        self.reward = 0.0  # reset reward each step
        day_cost = 0.0
        day_revenue = 0.0

        # 1. tick event countdowns (keep ticking into negative to track active duration)
        for event_name in self.events:
            self.events[event_name] -= 1

        total_inventory = sum(sum(batch[0] for batch in self.inventory[product]) for product in self.inventory)

        # 2. remove expired groceries
        new_batches = []
        expired_groceries_count = 0
        for batch in self.inventory["groceries"]:
            if batch[1] == 0:
                expired_groceries_count += batch[0]
                continue

            else:
                new_batches.append([batch[0], batch[1] - 1])

        self.inventory["groceries"] = new_batches

        # self.reward -= 0.05 * expired_groceries_count

        # 3. Handle incoming deliveries
        remaining_deliveries = []
        total_delivered_units = 0
        for delivery in self.deliveries:
            for product, shipment in delivery.items():
                qty, arrival_day = shipment
                if arrival_day <= self.current_day:
                    total_delivered_units += qty
                    self.inventory[product].append([qty, SHELF_LIFE[product]])
                else:
                    remaining_deliveries.append(delivery)
        self.deliveries = remaining_deliveries

        # 4. process purchases
        had_unaffordable_order = False
        for product, qty in action.buy_quantities.items():
            unit_cost = COST_PRICES[product] + SHIPPING_COST[action.delivery_method]
            total_cost = qty * unit_cost

            # capacity overage cost
            current_qty = sum(b[0] for b in self.inventory[product])
            overage = max(0, (current_qty + qty) - self.inventory_capacity[product])
            extra_cost = overage * EXTRA_INVENTORY_COST[product]
            total_cost += extra_cost

            if total_cost > self.cash:
                had_unaffordable_order = True
                # self.reward -= 0.5
                continue

            self.cash -= total_cost
            day_cost += total_cost

            arrival_day = self.current_day + SHIPPING_DAYS[action.delivery_method]
            # add jitter: slow ±2 days, medium ±1 day, fast is reliable
            jitter_rng = random.Random(self.seed * 2000 + self.current_day * 100 + hash(product))
            if action.delivery_method == "slow":
                arrival_day += jitter_rng.randint(-2, 2)
            elif action.delivery_method == "medium":
                arrival_day += jitter_rng.randint(-1, 1)
            # ensure arrival is at least next day
            arrival_day = max(self.current_day + 1, arrival_day)
            self.deliveries.append({product: [qty, arrival_day]})

        # 5. generate demand
        demand = self._generate_demand()

        # apply price elasticity: demand scales with price^(-elasticity)
        price_mults = {}
        for product in demand:
            pm = max(0.5, min(1.5, action.price_multipliers.get(product, 1.0)))
            price_mults[product] = pm
            e = PRICE_ELASTICITY[product]
            demand[product] = max(0, int(demand[product] * pm ** -e))

        # 6. sell products (fifo)
        max_daily_revenue = 0.0
        total_demand_units = 0
        total_sold_units = 0
        for product, demand_today in demand.items():

            sell_price = BASE_PRICES[product] * price_mults[product]
            max_daily_revenue += demand_today * sell_price
            product_availability = sum(batch[0] for batch in self.inventory[product])
            total_demand_units += demand_today


            if demand_today > product_availability:
                missed_sales = demand_today - product_availability
                sold = product_availability
                total_sold_units += sold
                day_revenue += sold * sell_price
                self.inventory[product] = []
                # self.reward -= missed_sales * sell_price * 0.001
                # self.reward += sold * sell_price * 0.001

            else:
                total_sold_units += demand_today
                day_revenue += demand_today * sell_price
                # self.reward += demand_today * sell_price * 0.001

                new_batches = []

                for batch in self.inventory[product]:
                    if batch[0] < demand_today:
                        demand_today = demand_today - batch[0]


                    elif demand_today == 0:
                        new_batches.append(batch)

                    else:
                        remaining = batch[0] - demand_today
                        if remaining > 0:
                            new_batches.append([remaining, batch[1]])
                        demand_today = 0

                self.inventory[product] = new_batches

        # 7. Liquidate some stock (FIFO, no revenue)
        total_liquidation_loss = 0.0
        total_liquidated_units = 0
        for product, count in action.liquidate.items():
            if product not in self.inventory or count <= 0:
                continue
            actually_removed = min(count, sum(b[0] for b in self.inventory[product]))
            total_liquidated_units += actually_removed
            total_liquidation_loss += actually_removed * COST_PRICES[product]
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

        # self.reward -= total_liquidation_loss * 0.001

        # compute day profit
        day_profit = day_revenue - day_cost
        self.cash += day_revenue
        self.total_profit += day_profit

        # check done
        done = self.current_day >= self.max_days

        # update state
        self._state = InventoryState(
            episode_id = self._state.episode_id,
            current_day = self.current_day,
            cash = self.cash,
            inventory = {p: sum(b[0] for b in self.inventory[p]) for p in self.inventory},
        )

        if day_profit < 0:
            R_profit_bool = -0.5
        elif day_profit == 0:
            R_profit_bool = 0.0 
        else:
            R_profit_bool = 1.0




        total_managed = total_inventory + total_delivered_units
        wasted_units = expired_groceries_count + total_liquidated_units
        waste_rate = wasted_units / max(total_managed, 1)
        R_waste = max(-1.0, 1.0 - 2.0 * min(waste_rate * 3, 1.0))

        # R_revenue: fraction of today's possible revenue captured
        R_revenue = max(-1.0, min(1.0, (2 * day_revenue / max(max_daily_revenue, 1.0) - 1.0)))

        # R_cash_health: sustainable cash position?
        cash_ratio = self.cash / self.task["initial_cash"]
        R_cash_health = max(-1.0, min(1.0, cash_ratio - 0.3))

        # R_capacity_util: warehouse utilization across all products
        # 0% util → -1.0, 50% → 0.0, 100% → +1.0
        utilizations = []
        for p in self.inventory:
            stock = sum(b[0] for b in self.inventory[p])
            utilizations.append(stock / self.inventory_capacity[p])
        avg_util = sum(utilizations) / len(utilizations)
        R_capacity_util = max(-1.0, min(1.0, 2.0 * avg_util - 1.0))

        # R_fulfillment: fraction of demand met (unit-based), mapped to [-1, +1]
        R_fulfillment = 2.0 * (total_sold_units / max(total_demand_units, 1)) - 1.0

        # Hard-fail: invalid action (negative quantities, invalid products, or liquidating more than available)
        invalid_action = False
        for p, qty in action.buy_quantities.items():
            if qty < 0 or p not in BASE_PRICES:
                invalid_action = True
                break
        for p, qty in action.liquidate.items():
            if qty < 0 or p not in BASE_PRICES:
                invalid_action = True
                break
            available = sum(b[0] for b in self.inventory.get(p, []))
            if qty > available:
                invalid_action = True
                break

        # Hard-fail: cash below $10
        bankrupt = self.cash < 10.0

        # Hard-fail: do-nothing for 3+ consecutive days
        is_idle = (not action.buy_quantities or all(v == 0 for v in action.buy_quantities.values())) and \
                  (not action.liquidate or all(v == 0 for v in action.liquidate.values()))
        if is_idle:
            self.consecutive_idle_days += 1
        else:
            self.consecutive_idle_days = 0
        idle_penalty = self.consecutive_idle_days >= 3

        # Apply hard-fail gates — independent, all stack on top of weighted reward
        hard_fail_penalty = 0.0
        if invalid_action:
            hard_fail_penalty -= 1.0
        if had_unaffordable_order:
            hard_fail_penalty -= 1.0
        if bankrupt:
            hard_fail_penalty -= 1.0
        if idle_penalty:
            hard_fail_penalty -= 1.0

        self.reward = (0.25 * R_revenue
                     + 0.20 * R_fulfillment
                     + 0.15 * R_waste
                     + 0.15 * R_cash_health
                     + 0.15 * R_capacity_util
                     + 0.10 * R_profit_bool
                     + hard_fail_penalty)

        return InventoryObservation(
            current_day = self.current_day,
            total_cash = self.cash,
            day_profit = day_profit,
            total_profit = self.total_profit,
            demand_today = demand,
            updated_inventory = copy.deepcopy(self.inventory),
            remaining_capacity = {p: max(0, self.inventory_capacity[p] - sum(b[0] for b in self.inventory[p])) for p in self.inventory},
            updated_events = copy.deepcopy(self.events),
            updated_deliveries = copy.deepcopy(self.deliveries),
            reward = self.reward,
            metadata = {
                "reward_breakdown": {
                    "R_profit_bool": R_profit_bool,
                    "R_revenue": R_revenue,
                    "R_fulfillment": R_fulfillment,
                    "R_waste": R_waste,
                    "R_cash_health": R_cash_health,
                    "R_capacity_util": R_capacity_util,
                },
                "hard_fails": {
                    "invalid_action": invalid_action,
                    "unaffordable_order": had_unaffordable_order,
                    "bankrupt": bankrupt,
                    "idle_penalty": idle_penalty,
                },
            },
            done = done,
        )


    def _generate_demand(self):
        rng = random.Random(self.seed * 1000 + self.current_day)
        demand = {}

        for product, (lo, hi) in self.base_demand.items():
            demand[product] = rng.randint(lo, hi)

        # weekend boost
        if self.current_day % 7 in (5, 6):
            for product in demand:
                demand[product] = int(demand[product] * WEEKEND_MULTIPLIER)

        # active event multipliers (only for EVENT_DURATION days after triggering)
        for event_name, days in self.events.items():
            if -EVENT_DURATION < days <= 0 and event_name in EVENT_EFFECTS:
                for product, mult in EVENT_EFFECTS[event_name].items():
                    demand[product] = int(demand[product] * mult)

        return demand


    @property
    def state(self) -> InventoryState:
        return self._state