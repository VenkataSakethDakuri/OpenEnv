from openenv.core.env_server.interfaces import Environment
import copy
import random
from uuid import uuid4

from models import InventoryAction, InventoryObservation, InventoryState
from .constants import (
    INITIAL_CASH, BASE_PRICES, COST_PRICES, SHELF_LIFE, INITIAL_STOCK,
    EVENTS, SHIPPING_COST, SHIPPING_DAYS, INVENTORY_CAPACITY,
    EXTRA_INVENTORY_COST, BASE_DEMAND, WEEKEND_MULTIPLIER, EVENT_EFFECTS,
    EVENT_DURATION, MAX_DAYS, UPGRADE_DELIVERY_COST, TASKS,
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

        self.reward -= 0.05 * expired_groceries_count

        # 3. Handle incoming deliveries
        remaining_deliveries = []
        for delivery in self.deliveries:
            for product, shipment in delivery.items():
                qty, arrival_day = shipment
                if arrival_day <= self.current_day:
                    self.inventory[product].append([qty, SHELF_LIFE[product]])
                else:
                    remaining_deliveries.append(delivery)
        self.deliveries = remaining_deliveries

        # 4. process purchases
        for product, qty in action.buy_quantities.items():
            unit_cost = COST_PRICES[product] + SHIPPING_COST[action.delivery_method]
            total_cost = qty * unit_cost

            # capacity overage cost
            current_qty = sum(b[0] for b in self.inventory[product])
            overage = max(0, (current_qty + qty) - self.inventory_capacity[product])
            extra_cost = overage * EXTRA_INVENTORY_COST[product]
            total_cost += extra_cost

            if total_cost > self.cash:
                self.reward -= 0.5  # penalize for ordering what you can't afford
                continue

            self.cash -= total_cost
            day_cost += total_cost

            arrival_day = self.current_day + SHIPPING_DAYS[action.delivery_method]
            self.deliveries.append({product: [qty, arrival_day]})

        # 5. generate demand
        demand = self._generate_demand()

        # 6. sell products (fifo)
        for product, demand_today in demand.items():

            product_availability = sum(batch[0] for batch in self.inventory[product])


            if demand_today > product_availability:
                missed_sales = demand_today - product_availability
                sold = product_availability
                day_revenue += sold * BASE_PRICES[product]
                self.inventory[product] = []
                self.reward -= missed_sales * BASE_PRICES[product] * 0.001
                self.reward += sold * BASE_PRICES[product] * 0.001

            else:
                day_revenue += demand_today * BASE_PRICES[product]
                self.reward += demand_today * BASE_PRICES[product] * 0.001

                new_batches = []

                for batch in self.inventory[product]:
                    if batch[0] < demand_today:
                        demand_today = demand_today - batch[0]


                    elif demand_today == 0:
                        new_batches.append(batch)

                    else:
                        new_batches.append([batch[0] - demand_today, batch[1]])
                        demand_today = 0

                self.inventory[product] = new_batches

        # 7. Liquidate some stock (FIFO, no revenue)
        total_liquidation_loss = 0.0
        for product, count in action.liquidate.items():
            if product not in self.inventory or count <= 0:
                continue
            actually_removed = min(count, sum(b[0] for b in self.inventory[product]))
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

        self.reward -= total_liquidation_loss * 0.001

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