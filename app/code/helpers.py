"""Pure, Streamlit-free helpers for the car-price category wizard.

Kept free of any ``streamlit`` import so the logic (skip handling, rupee
formatting, class copy) is unit-testable without a UI. ``ui.py`` builds the
widgets on top of the data defined here.
"""

# Sentinel shown as the first option of every dropdown; choosing it = "skip".
NOT_SURE = "🤷 Not sure"

BRANDS = ["Maruti", "Hyundai", "Mahindra", "Tata", "Toyota", "Honda", "Ford",
          "Chevrolet", "Renault", "Volkswagen", "BMW", "Skoda", "Nissan",
          "Jaguar", "Volvo", "Datsun", "Mercedes-Benz", "Fiat", "Audi", "Other"]

OWNER_LABELS = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]

# One-line description shown with each predicted class.
CLASS_BLURB = {
    "Budget": "Entry-level cars — the most affordable quarter of the market.",
    "Mid-range": "Solid everyday cars priced in the lower-middle band.",
    "Premium": "Higher-spec or newer cars in the upper-middle band.",
    "Luxury": "The most expensive quarter — premium brands and near-new cars.",
}

# Numeric fields:  field -> (label, emoji, min, max, default, step)
NUMERIC_SPECS = {
    "year":      ("Year", "📅", 1990, 2024, 2015, 1),
    "km_driven": ("Kilometres driven", "🛣️", 0, 1_000_000, 50_000, 1_000),
    "mileage":   ("Mileage (kmpl)", "⛽", 0.0, 50.0, 20.0, 0.5),
    "engine":    ("Engine (CC)", "🔧", 600.0, 5000.0, 1200.0, 50.0),
    "max_power": ("Max power (bhp)", "💪", 20.0, 500.0, 85.0, 1.0),
    "seats":     ("Seats", "💺", 2, 14, 5, 1),
}

# Dropdown fields:  field -> (label, emoji, options)
SELECT_SPECS = {
    "brand":        ("Brand", "🚘", BRANDS),
    "fuel":         ("Fuel", "⛽", ["Petrol", "Diesel"]),
    "transmission": ("Transmission", "⚙️", ["Manual", "Automatic"]),
    "owner":        ("Owner", "👤", OWNER_LABELS),
    "seller_type":  ("Seller type", "🏷️", ["Individual", "Dealer", "Trustmark Dealer"]),
}

# Which fields belong to each wizard step.
STEP_BASICS = ["brand", "year", "fuel", "transmission", "km_driven"]
STEP_DETAILS = ["owner", "seller_type", "mileage", "engine", "max_power", "seats"]


def resolve_numeric(skipped: bool, value):
    """A skipped numeric field becomes None (→ NaN → imputed); else the value."""
    return None if skipped else value


def resolve_choice(choice):
    """A dropdown left unset or on the 'not sure' sentinel becomes None."""
    return None if choice is None or choice == NOT_SURE else choice


def format_inr(amount) -> str:
    """Format a number as Indian-grouped rupees, e.g. 485000 -> '₹4,85,000'."""
    n = int(round(amount))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        body = s
    else:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:                 # group all but the last 3 digits in 2s
            groups.insert(0, head[-2:])
            head = head[:-2]
        groups.insert(0, head)
        body = ",".join(groups) + "," + tail
    return f"₹{sign}{body}"


def format_range(low, high) -> str:
    """Human range string for a price class, e.g. '₹2,50,000 – ₹4,10,000'."""
    if high is None:
        return f"{format_inr(low)} and above"
    return f"{format_inr(low)} – {format_inr(high)}"
