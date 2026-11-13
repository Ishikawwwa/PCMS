"""Disclosed raw development cases. Shopping uses a partial published trace."""

from .live_memory import Evidence


SHOPPING = (
    Evidence("shopping-budget", "user", "The total budget for Bundle 24 is $100.00."),
    Evidence("shopping-1", "tool", "Purchase confirmed for Bundle 24, Product 1: Gluten-free carrot cake mix (B00T6NA7PA), $14.99."),
    Evidence("shopping-2", "tool", "Purchase confirmed for Bundle 24, Product 2: Betty Crocker ready-to-serve cream cheese frosting, 8 pack (B003GQP06O), $34.27."),
    Evidence("shopping-search", "tool", "Bundle 24 search results included a Mermaid Cake Cup (B084JBT37C) for $6.99. It was considered and never purchased."),
    Evidence("shopping-3", "tool", "Purchase confirmed for Bundle 24, Product 3: AmeriColor Tulip Red airbrush food color (B071S91WRZ), $16.00."),
)
SHOPPING_QUERY = "Buy Product 4 for Bundle 24. Use the confirmed predecessor to apply compatibility rules and stay within the remaining bundle budget."

TRAVEL = (
    Evidence("travel-opening", "user", "For Trip 1, we confirmed Rockford as the Day 2 destination. Eric confirmed he needs step-free transport on Day 2."),
    Evidence("travel-emma", "user", "Emma joined Trip 1. She confirmed a Day 2 transport limit of $80 per person."),
    Evidence("travel-other-emma", "user", "On a different trip, Trip 2, a different traveler named Emma confirmed a transport limit of $150 per person."),
    Evidence("travel-proposal", "user", "Someone suggested increasing Emma's Day 2 transport limit on Trip 1 to $100 per person. Emma has not agreed."),
    Evidence("travel-confirmation", "user", "Emma confirmed her Day 2 transport limit on Trip 1 is now $100 per person, replacing her earlier $80 limit."),
)
TRAVEL_QUERY = "Choose transport for the group on Day 2 of Trip 1. Choose the fastest offer that meets every applicable constraint."
TRAVEL_OFFERS = (
    {"id": "rail", "destination": "Rockford", "minutes": 30, "step_free": True, "per_person_dollars": 95},
    {"id": "bus", "destination": "Rockford", "minutes": 50, "step_free": True, "per_person_dollars": 75},
    {"id": "van", "destination": "Rockford", "minutes": 25, "step_free": False, "per_person_dollars": 60},
    {"id": "express", "destination": "Oakland", "minutes": 20, "step_free": True, "per_person_dollars": 55},
)
