INITIAL_CASH = 1000.0

# Product name -> base price (selling price before multiplier)
BASE_PRICES = {
    "electronics": 150.0,
    "clothing": 40.0,
    "groceries": 10.0,
    "furniture": 200.0,
    "toys": 25.0,
}

# Product name -> cost price (what you pay to buy stock)
COST_PRICES = {
    "electronics": 100.0,
    "clothing": 25.0,
    "groceries": 5.0,
    "furniture": 130.0,
    "toys": 12.0,
}

# Product name -> shelf life in days (None = no expiry)
SHELF_LIFE = {
    "electronics": None,
    "clothing": None,
    "groceries": 5,
    "furniture": None,
    "toys": None,
}

# Product name -> starting stock quantity
INITIAL_STOCK = {
    "electronics": 10,
    "clothing": 20,
    "groceries": 50,
    "furniture": 5,
    "toys": 30,
}

# Delivery method -> cost per unit
SHIPPING_COST = {
    "slow": 2.0,
    "medium": 5.0,
    "fast": 10.0,
}

# Delivery method -> days to arrive
SHIPPING_DAYS = {
    "slow": 5,
    "medium": 3,
    "fast": 1,
}

# Event name -> days until event (spread across 30 days)
EVENTS = {
    "black_friday": 6,
    "christmas": 12,
    "back_to_school": 18,
    "summer_clearance": 24,
    "new_competitor": 28,
}

# Product name -> max inventory space (units)
INVENTORY_CAPACITY = {
    "electronics": 100,
    "clothing": 200,
    "groceries": 500,
    "furniture": 50,
    "toys": 300,
}

# Product name -> additional cost per unit for extra inventory beyond capacity
EXTRA_INVENTORY_COST = {
    "electronics": 20.0,
    "clothing": 5.0,
    "groceries": 2.0,
    "furniture": 30.0,
    "toys": 4.0,
}

# Product name -> (min_demand, max_demand) per day
BASE_DEMAND = {
    "electronics": (3, 8),
    "clothing": (5, 15),
    "groceries": (20, 40),
    "furniture": (1, 3),
    "toys": (5, 12),
}

WEEKEND_MULTIPLIER = 1.2

# Event name -> {product: demand_multiplier} when event triggers
EVENT_EFFECTS = {
    "black_friday": {"electronics": 3.0, "clothing": 2.5, "toys": 2.0, "furniture": 1.5, "groceries": 1.0},
    "christmas": {"toys": 3.0, "electronics": 2.0, "clothing": 1.5, "furniture": 1.0, "groceries": 1.5},
    "back_to_school": {"clothing": 2.5, "electronics": 1.5, "toys": 1.5, "furniture": 1.0, "groceries": 1.0},
    "summer_clearance": {"clothing": 2.0, "toys": 1.5, "electronics": 1.0, "furniture": 1.5, "groceries": 1.0},
    "new_competitor": {"electronics": 0.6, "clothing": 0.7, "toys": 0.7, "furniture": 0.8, "groceries": 0.9},
}

EVENT_DURATION = 2

MAX_DAYS = 30

UPGRADE_DELIVERY_COST = 50.0

# Task configs for easy/medium/hard
TASKS = {
    # Easy: High starting stock, low demand, no events, full warehouse capacity.
    # Agent just needs to maintain stock and sell. Minimal challenge.
    "easy": {
        "seed": 100,
        "max_days": 30,
        "initial_cash": 1000.0,
        "events": {},  # no events
        "initial_stock": {
            "electronics": 5,
            "clothing": 10,
            "groceries": 20,
            "furniture": 3,
            "toys": 10,
        },
        "inventory_capacity": INVENTORY_CAPACITY,
        "base_demand": {
            "electronics": (2, 5),
            "clothing": (3, 10),
            "groceries": (15, 30),
            "furniture": (1, 2),
            "toys": (3, 8),
        },
    },
    # Medium: Default stock/cash, all 5 events spread across 30 days, normal demand.
    # Agent must anticipate demand spikes from events and restock accordingly.
    "medium": {
        "seed": 200,
        "max_days": 30,
        "initial_cash": 1000.0,
        "events": EVENTS,
        "initial_stock": INITIAL_STOCK,
        "inventory_capacity": INVENTORY_CAPACITY,
        "base_demand": BASE_DEMAND,
    },
    # Hard: Half starting cash ($500), low stock, events packed close together,
    # higher demand, smaller warehouse. Agent must balance tight budget,
    # overlapping event spikes, and fast-expiring groceries.
    "hard": {
        "seed": 300,
        "max_days": 30,
        "initial_cash": 500.0,
        "events": {
            "black_friday": 4,
            "christmas": 8,
            "back_to_school": 12,
            "summer_clearance": 16,
            "new_competitor": 20,
        },
        "initial_stock": {
            "electronics": 5,
            "clothing": 10,
            "groceries": 30,
            "furniture": 3,
            "toys": 15,
        },
        "inventory_capacity": {
            "electronics": 50,
            "clothing": 100,
            "groceries": 250,
            "furniture": 25,
            "toys": 150,
        },
        "base_demand": {
            "electronics": (5, 12),
            "clothing": (8, 20),
            "groceries": (30, 60),
            "furniture": (2, 5),
            "toys": (8, 18),
        },
    },
}