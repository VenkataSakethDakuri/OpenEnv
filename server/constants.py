MAX_DAYS = 90

BASE_PRICES = {
    "electronics": 150.0,
    "clothing": 40.0,
    "groceries": 10.0,
    "furniture": 200.0,
    "toys": 25.0,
}

COST_PRICES = {
    "electronics": 100.0,
    "clothing": 25.0,
    "groceries": 5.0,
    "furniture": 130.0,
    "toys": 12.0,
}

SHELF_LIFE = {
    "electronics": None,
    "clothing": None,
    "groceries": 5,
    "furniture": None,
    "toys": None,
}

INITIAL_STOCK = {
    "electronics": 10,
    "clothing": 20,
    "groceries": 50,
    "furniture": 5,
    "toys": 30,
}

SHIPPING_COST = {"slow": 2.0, "medium": 5.0, "fast": 10.0}
SHIPPING_DAYS = {"slow": 5, "medium": 3, "fast": 1}

INVENTORY_CAPACITY = {
    "electronics": 100,
    "clothing": 200,
    "groceries": 500,
    "furniture": 50,
    "toys": 300,
}

EXTRA_INVENTORY_COST = {
    "electronics": 20.0,
    "clothing": 5.0,
    "groceries": 2.0,
    "furniture": 30.0,
    "toys": 4.0,
}

BASE_DEMAND = {
    "electronics": (3, 8),
    "clothing": (5, 15),
    "groceries": (20, 40),
    "furniture": (1, 3),
    "toys": (5, 12),
}

PRICE_ELASTICITY = {
    "electronics": 1.2,
    "clothing": 1.5,
    "groceries": 0.4,
    "furniture": 0.8,
    "toys": 1.3,
}

WEEKEND_MULTIPLIER = 1.2
EVENT_DURATION = 3

# --- EVENTS ---

EVENTS_EASY = {}

EVENTS_MEDIUM = {
    "valentines": 10,
    "spring_sale": 25,
    "memorial_day": 40,
    "summer_clearance": 55,
    "back_to_school": 70,
    "labor_day": 82,
}

EVENTS_HARD = {
    "valentines": 8,
    "flash_sale_1": 15,
    "spring_sale": 22,
    "supply_disruption": 30,
    "memorial_day": 38,
    "competitor_launch": 45,
    "summer_clearance": 52,
    "flash_sale_2": 60,
    "back_to_school": 68,
    "labor_day": 75,
    "early_holiday": 82,
    "black_friday_preview": 87,
}

EVENT_EFFECTS = {
    "valentines": {"toys": 2.0, "clothing": 1.8, "electronics": 1.3, "furniture": 1.0, "groceries": 1.2},
    "spring_sale": {"clothing": 2.5, "furniture": 2.0, "toys": 1.5, "electronics": 1.2, "groceries": 1.0},
    "memorial_day": {"groceries": 2.0, "furniture": 1.5, "electronics": 1.3, "clothing": 1.0, "toys": 1.0},
    "summer_clearance": {"clothing": 2.0, "toys": 1.5, "electronics": 1.0, "furniture": 1.5, "groceries": 1.0},
    "back_to_school": {"electronics": 2.5, "clothing": 2.0, "toys": 1.5, "furniture": 1.0, "groceries": 1.0},
    "labor_day": {"groceries": 2.0, "electronics": 1.5, "clothing": 1.3, "furniture": 1.5, "toys": 1.0},
    "flash_sale_1": {"electronics": 3.0, "toys": 2.5, "clothing": 2.0, "furniture": 1.5, "groceries": 1.0},
    "flash_sale_2": {"electronics": 2.5, "clothing": 3.0, "toys": 2.0, "furniture": 1.0, "groceries": 1.5},
    "supply_disruption": {"electronics": 0.5, "furniture": 0.4, "clothing": 0.7, "toys": 0.8, "groceries": 1.0},
    "competitor_launch": {"electronics": 0.6, "toys": 0.7, "clothing": 0.8, "furniture": 0.9, "groceries": 1.0},
    "early_holiday": {"toys": 3.0, "electronics": 2.0, "clothing": 1.5, "furniture": 1.0, "groceries": 1.5},
    "black_friday_preview": {"electronics": 3.5, "clothing": 2.5, "toys": 3.0, "furniture": 2.0, "groceries": 1.0},
}

# --- DIRECTIVES ---

DIRECTIVES_EASY = [
    {"day": 5, "id": "E01", "type": "min_stock",
     "text": "Maintain at least 10 units of each product in stock at all times.",
     "params": {"product": "all", "min_qty": 10}, "penalty": -0.3, "expires": None, "modifies": None},
    {"day": 10, "id": "E02", "type": "shipping_rule",
     "text": "Use medium or fast shipping for all grocery orders to reduce spoilage.",
     "params": {"product": "groceries", "allowed_methods": ["medium", "fast"]},
     "penalty": -0.5, "expires": None, "modifies": None},
    {"day": 25, "id": "E03", "type": "budget_cap",
     "text": "Keep daily spending below $350 for the remainder of the quarter.",
     "params": {"period": "daily", "max_amount": 350}, "penalty": -0.5, "expires": None, "modifies": None},
    {"day": 50, "id": "E04", "type": "price_range",
     "text": "Do not raise grocery prices above 1.2x. Customer satisfaction policy.",
     "params": {"product": "groceries", "min_mult": 0.5, "max_mult": 1.2},
     "penalty": -0.5, "expires": None, "modifies": None},
    {"day": 75, "id": "E05", "type": "waste_limit",
     "text": "Reduce waste: weekly waste must stay below 20 units for the final stretch.",
     "params": {"max_units": 20}, "penalty": -0.5, "expires": None, "modifies": None},
]

DIRECTIVES_MEDIUM = [
    {"day": 3, "id": "M01", "type": "shipping_rule",
     "text": "All grocery orders must use fast shipping effective immediately.",
     "params": {"product": "groceries", "allowed_methods": ["fast"]},
     "penalty": -1.0, "expires": None, "modifies": None},
    {"day": 7, "id": "M02", "type": "min_stock",
     "text": "Maintain electronics stock above 20 units at all times.",
     "params": {"product": "electronics", "min_qty": 20}, "penalty": -0.5, "expires": None, "modifies": None},
    {"day": 12, "id": "M03", "type": "budget_cap",
     "text": "Weekly shipping budget capped at $500. Effective next Monday.",
     "params": {"period": "weekly", "max_amount": 500}, "penalty": -1.0, "expires": None, "modifies": None},
    {"day": 25, "id": "M05", "type": "min_stock",
     "text": "Prepare for spring sale: maintain clothing stock above 40 units.",
     "params": {"product": "clothing", "min_qty": 40}, "penalty": -1.0, "expires": 30, "modifies": None},
    {"day": 30, "id": "M06", "type": "shipping_rule",
     "text": "UPDATE to M01: Grocery fast-shipping requirement relaxed. Medium shipping now acceptable.",
     "params": {"product": "groceries", "allowed_methods": ["medium", "fast"]},
     "penalty": -0.5, "expires": None, "modifies": "M01"},
    {"day": 35, "id": "M07", "type": "waste_limit",
     "text": "New policy: weekly waste must stay below 15 units (expired + liquidated).",
     "params": {"max_units": 15}, "penalty": -1.0, "expires": None, "modifies": None},
    {"day": 40, "id": "M08", "type": "price_range",
     "text": "Do not discount electronics below 0.9x price. Brand positioning policy.",
     "params": {"product": "electronics", "min_mult": 0.9, "max_mult": 1.5},
     "penalty": -1.5, "expires": None, "modifies": None},
    {"day": 45, "id": "M09", "type": "order_limit",
     "text": "Logistics constraint: order a maximum of 4 different products per day.",
     "params": {"max_products": 4}, "penalty": -0.5, "expires": None, "modifies": None},
    # DECEPTIVE: Looks like an opportunity but groceries expire in 5 days.
    # Stocking 150 when demand is ~20-40/day means ~70-110 will spoil.
    # Waste limit M07 is active (15 units/week, penalty -1.0/step).
    # Smart agent: ignore this (-0.5 one-time) rather than trigger waste violations (-1.0/step).
    {"day": 47, "id": "M16", "type": "target_stock",
     "text": "OPPORTUNITY: Major grocery demand surge expected! Stock 150+ groceries by day 55 to maximize revenue.",
     "params": {"product": "groceries", "min_qty": 150, "deadline": 55}, "penalty": -0.5, "expires": 55, "modifies": None},
    {"day": 52, "id": "M10", "type": "min_stock",
     "text": "Summer clearance prep: maintain toys stock above 30 units.",
     "params": {"product": "toys", "min_qty": 30}, "penalty": -1.0, "expires": 60, "modifies": None},
    {"day": 58, "id": "M11", "type": "price_range",
     "text": "Clearance pricing: clothing must be priced at 0.7x-0.9x to move old stock.",
     "params": {"product": "clothing", "min_mult": 0.7, "max_mult": 0.9},
     "penalty": -1.5, "expires": 65, "modifies": None},
    {"day": 63, "id": "M12", "type": "budget_cap",
     "text": "Cost reduction: limit daily total spending to $400 until end of quarter.",
     "params": {"period": "daily", "max_amount": 400}, "penalty": -0.5, "expires": None, "modifies": None},
    {"day": 70, "id": "M13", "type": "min_stock",
     "text": "Stock up for back-to-school: maintain 50+ electronics at all times.",
     "params": {"product": "electronics", "min_qty": 50}, "penalty": -2.0, "expires": None, "modifies": "M02"},
    {"day": 78, "id": "M14", "type": "budget_cap",
     "text": "UPDATE to M03: Weekly shipping budget increased to $700 for final stretch.",
     "params": {"period": "weekly", "max_amount": 700}, "penalty": -1.0, "expires": None, "modifies": "M03"},
    {"day": 85, "id": "M15", "type": "order_limit",
     "text": "Wind-down approaching: limit orders to max 3 products per day.",
     "params": {"max_products": 3}, "penalty": -1.0, "expires": None, "modifies": "M09"},
]

DIRECTIVES_HARD = [
    {"day": 2, "id": "H01", "type": "shipping_rule",
     "text": "All grocery orders must use fast shipping. No exceptions.",
     "params": {"product": "groceries", "allowed_methods": ["fast"]},
     "penalty": -1.0, "expires": None, "modifies": None},
    {"day": 4, "id": "H02", "type": "min_stock",
     "text": "Never let any product drop below 5 units. Maintain minimum stock at all times.",
     "params": {"product": "all", "min_qty": 5}, "penalty": -0.8, "expires": None, "modifies": None},
    {"day": 6, "id": "H03", "type": "budget_cap",
     "text": "Daily total spending (orders + shipping) must not exceed $400.",
     "params": {"period": "daily", "max_amount": 400}, "penalty": -1.0, "expires": None, "modifies": None},
    {"day": 9, "id": "H04", "type": "price_range",
     "text": "Toys price multiplier must stay between 0.8 and 1.2 at all times.",
     "params": {"product": "toys", "min_mult": 0.8, "max_mult": 1.2},
     "penalty": -0.5, "expires": None, "modifies": None},
    {"day": 14, "id": "H06", "type": "shipping_rule",
     "text": "Electronics orders must use medium or fast shipping to avoid delays.",
     "params": {"product": "electronics", "allowed_methods": ["medium", "fast"]},
     "penalty": -1.0, "expires": 25, "modifies": None},
    {"day": 17, "id": "H07", "type": "waste_limit",
     "text": "Zero tolerance: no more than 10 units may be wasted per week.",
     "params": {"max_units": 10}, "penalty": -1.5, "expires": None, "modifies": None},
    {"day": 20, "id": "H08", "type": "budget_cap",
     "text": "CORRECTION to H03: Daily spending limit increased to $500 due to upcoming events.",
     "params": {"period": "daily", "max_amount": 500}, "penalty": -1.0, "expires": None, "modifies": "H03"},
    {"day": 23, "id": "H09", "type": "force_liquidate",
     "text": "URGENT: Product safety recall on furniture. Liquidate ALL furniture by day 26.",
     "params": {"product": "furniture", "deadline": 26}, "penalty": -5.0, "expires": 26, "modifies": None},
    {"day": 30, "id": "H11", "type": "min_stock",
     "text": "Month 2 readiness: maintain at least 15 units of every product.",
     "params": {"product": "all", "min_qty": 15}, "penalty": -1.5, "expires": None, "modifies": "H02"},
    # CONFLICT: Order freeze vs H11 min_stock (15 of all, penalty -1.5/step).
    # As demand depletes stock, agent can't restock. Must choose:
    # Violate freeze (-1.0/step) to maintain stock, or let stock drop (-1.5/step).
    # Optimal: violate freeze (lower penalty) and keep ordering.
    {"day": 34, "id": "H29", "type": "order_freeze",
     "text": "Inventory reduction initiative: freeze ALL new orders effective immediately until day 42.",
     "params": {"after_day": 33}, "penalty": -1.0, "expires": 42, "modifies": None},
    {"day": 37, "id": "H13", "type": "min_cash",
     "text": "Cash reserve policy: maintain cash above $300 at end of each day.",
     "params": {"min_amount": 300},
     "penalty": -2.0, "expires": None, "modifies": None},
    {"day": 40, "id": "H14", "type": "order_limit",
     "text": "Diversity rule: when ordering, order at least 2 and no more than 4 different products.",
     "params": {"min_products": 2, "max_products": 4}, "penalty": -1.0, "expires": None, "modifies": None},
    {"day": 44, "id": "H15", "type": "price_range",
     "text": "Electronics premium positioning: price at 1.2x or higher during days 45-55.",
     "params": {"product": "electronics", "min_mult": 1.2, "max_mult": 1.5},
     "penalty": -1.5, "expires": 55, "modifies": None},
    {"day": 48, "id": "H16", "type": "price_range",
     "text": "Clearance: clothing must be priced 0.6x-0.8x to move old inventory before events.",
     "params": {"product": "clothing", "min_mult": 0.6, "max_mult": 0.8},
     "penalty": -1.5, "expires": 55, "modifies": None},
    # DECEPTIVE: Groceries expire in 5 days. Stocking 200 when demand is ~25-50/day
    # means ~50-100 will spoil. H18 waste limit (20 units/week, -1.0) will trigger
    # repeatedly. Smart agent: take the -0.5 one-time penalty rather than trigger waste cascade.
    {"day": 50, "id": "H30", "type": "target_stock",
     "text": "URGENT: Summer clearance grocery push! Stock 200+ groceries by day 56 for maximum sales volume.",
     "params": {"product": "groceries", "min_qty": 200, "deadline": 56}, "penalty": -0.5, "expires": 56, "modifies": None},
    {"day": 52, "id": "H17", "type": "force_liquidate",
     "text": "Clothing contamination: liquidate all clothing stock by day 55.",
     "params": {"product": "clothing", "deadline": 55}, "penalty": -4.0, "expires": 55, "modifies": None},
    {"day": 56, "id": "H18", "type": "waste_limit",
     "text": "UPDATE to H07: Waste limit relaxed to 20 units per week.",
     "params": {"max_units": 20}, "penalty": -1.0, "expires": None, "modifies": "H07"},
    {"day": 60, "id": "H19", "type": "shipping_rule",
     "text": "Quality handling: furniture orders must use medium or fast shipping to prevent transit damage.",
     "params": {"product": "furniture", "allowed_methods": ["medium", "fast"]},
     "penalty": -1.0, "expires": 70, "modifies": None},
    {"day": 63, "id": "H20", "type": "budget_cap",
     "text": "Final month austerity: daily spending capped at $350.",
     "params": {"period": "daily", "max_amount": 350}, "penalty": -1.0, "expires": None, "modifies": "H08"},
    {"day": 66, "id": "H21", "type": "waste_limit",
     "text": "Green initiative: zero grocery waste allowed for days 66-80. Max 0 units.",
     "params": {"max_units": 0}, "penalty": -2.0, "expires": 80, "modifies": "H18"},
    # CONFLICT: Order freeze (days 69-75, penalty -1.0) vs H23 on day 73
    # (maintain 80+ electronics, penalty -2.0). Agent can't stock electronics
    # while orders are frozen. Must violate freeze (-1.0) to satisfy H23 (-2.0).
    # Optimal: break the freeze and buy electronics.
    {"day": 69, "id": "H31", "type": "order_freeze",
     "text": "Cost containment: freeze ALL new orders effective immediately until day 75.",
     "params": {"after_day": 68}, "penalty": -1.0, "expires": 75, "modifies": None},
    {"day": 73, "id": "H23", "type": "min_stock",
     "text": "Stock up: maintain 80+ electronics by day 78 for upcoming sale.",
     "params": {"product": "electronics", "min_qty": 80}, "penalty": -2.0, "expires": 78, "modifies": None},
    {"day": 76, "id": "H24", "type": "price_range",
     "text": "CANCEL electronics premium pricing. Return to normal range 0.8-1.5.",
     "params": {"product": "electronics", "min_mult": 0.8, "max_mult": 1.5},
     "penalty": -0.5, "expires": None, "modifies": "H15"},
    {"day": 79, "id": "H25", "type": "min_stock",
     "text": "Flash demand incoming: maintain 100+ toys for days 80-83.",
     "params": {"product": "toys", "min_qty": 100}, "penalty": -5.0, "expires": 83, "modifies": None},
    {"day": 82, "id": "H26", "type": "budget_cap",
     "text": "Override austerity: spend limit raised to $600/day for final push.",
     "params": {"period": "daily", "max_amount": 600}, "penalty": -0.5, "expires": None, "modifies": "H20"},
    {"day": 85, "id": "H27", "type": "waste_limit",
     "text": "Final stretch: zero waste allowed in last 5 days. Max 0 units wasted per week.",
     "params": {"max_units": 0}, "penalty": -3.0, "expires": None, "modifies": "H18"},
    {"day": 88, "id": "H28", "type": "order_freeze",
     "text": "Wind-down: do NOT place any new orders after today. Sell remaining stock only.",
     "params": {"after_day": 88}, "penalty": -3.0, "expires": None, "modifies": None},
]

# --- MILESTONES ---

MILESTONES_EASY = {
    "early_profit": {"target": 300, "metric": "total_profit", "deadline": 20, "bonus": 1.5},
    "profit_600": {"target": 600, "metric": "total_profit", "deadline": 40, "bonus": 2.0},
    "low_waste": {"target": 0.10, "metric": "waste_rate_below", "deadline": 50, "bonus": 2.0},
    "profit_1200": {"target": 1200, "metric": "total_profit", "deadline": 70, "bonus": 3.0},
    "profit_1800": {"target": 1800, "metric": "total_profit", "deadline": 90, "bonus": 5.0},
}

MILESTONES_MEDIUM = {
    "week2_profit": {"target": 400, "metric": "total_profit", "deadline": 15, "bonus": 1.5},
    "month1": {"target": 900, "metric": "total_profit", "deadline": 30, "bonus": 2.0},
    "low_waste": {"target": 0.08, "metric": "waste_rate_below", "deadline": 45, "bonus": 2.0},
    "month2": {"target": 2800, "metric": "total_profit", "deadline": 60, "bonus": 3.0},
    "profit_4000": {"target": 4000, "metric": "total_profit", "deadline": 75, "bonus": 3.0},
    "final": {"target": 5500, "metric": "total_profit", "deadline": 90, "bonus": 5.0},
}

MILESTONES_HARD = {
    "first_profit": {"target": 200, "metric": "total_profit", "deadline": 15, "bonus": 1.5},
    "survive": {"target": 600, "metric": "total_profit", "deadline": 25, "bonus": 2.0},
    "recall_done": {"target": 0, "metric": "furniture_stock_zero", "deadline": 26, "bonus": 3.0},
    "month1": {"target": 800, "metric": "total_profit", "deadline": 30, "bonus": 2.0},
    "midpoint": {"target": 2000, "metric": "total_profit", "deadline": 50, "bonus": 2.0},
    "month2": {"target": 3800, "metric": "total_profit", "deadline": 60, "bonus": 3.0},
    "zero_waste": {"target": 14, "metric": "grocery_waste_zero_streak", "deadline": 80, "bonus": 4.0},
    "flash_ready": {"target": 100, "metric": "toys_stock_above", "deadline": 79, "bonus": 3.0},
    "profit_5500": {"target": 5500, "metric": "total_profit", "deadline": 80, "bonus": 3.0},
    "final": {"target": 7000, "metric": "total_profit", "deadline": 90, "bonus": 5.0},
}

# --- TASK CONFIGS ---

TASKS = {
    "easy": {
        "seed": 100,
        "max_days": 90,
        "initial_cash": 2000.0,
        "events": EVENTS_EASY,
        "directives": DIRECTIVES_EASY,
        "milestones": MILESTONES_EASY,
        "initial_stock": {
            "electronics": 15, "clothing": 30, "groceries": 60, "furniture": 8, "toys": 40,
        },
        "inventory_capacity": INVENTORY_CAPACITY,
        "base_demand": {
            "electronics": (2, 5), "clothing": (3, 10), "groceries": (15, 30),
            "furniture": (1, 2), "toys": (3, 8),
        },
    },
    "medium": {
        "seed": 200,
        "max_days": 90,
        "initial_cash": 1500.0,
        "events": EVENTS_MEDIUM,
        "directives": DIRECTIVES_MEDIUM,
        "milestones": MILESTONES_MEDIUM,
        "initial_stock": {
            "electronics": 10, "clothing": 20, "groceries": 50, "furniture": 5, "toys": 30,
        },
        "inventory_capacity": INVENTORY_CAPACITY,
        "base_demand": BASE_DEMAND,
    },
    "hard": {
        "seed": 300,
        "max_days": 90,
        "initial_cash": 1000.0,
        "events": EVENTS_HARD,
        "directives": DIRECTIVES_HARD,
        "milestones": MILESTONES_HARD,
        "initial_stock": {
            "electronics": 8, "clothing": 15, "groceries": 40, "furniture": 5, "toys": 20,
        },
        "inventory_capacity": {p: int(v * 0.8) for p, v in INVENTORY_CAPACITY.items()},
        "base_demand": {
            "electronics": (5, 12), "clothing": (8, 20), "groceries": (25, 50),
            "furniture": (2, 4), "toys": (6, 15),
        },
    },
}