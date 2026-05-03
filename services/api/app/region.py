"""Regional configuration — Thailand-first.

Scopes the product to Thailand to avoid huge global datasets and to
tune geocoding, pricing, VLM prompts, and test fixtures to the local
market (corrugated zinc/metal roofs, monsoon climate, THB currency).

Override with REGION env var to enable other markets later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════
# Thailand bounding box (approximate)
# ═══════════════════════════════════════════════════════════════
# Covers mainland Thailand + southern provinces + northern hills.
# Used for geocoder biasing and to reject out-of-region queries.
THAILAND_BBOX = {
    "min_lat": 5.6,     # southern tip (Betong/Narathiwat)
    "max_lat": 20.5,    # northern border (Mae Sai)
    "min_lng": 97.3,    # western border (Mae Hong Son)
    "max_lng": 105.7,   # eastern border (Ubon Ratchathani)
}

# Popular city centers for default/test cases
THAI_CITY_CENTERS = {
    "bangkok": (13.7563, 100.5018),
    "chiang_mai": (18.7883, 98.9853),
    "phuket": (7.8804, 98.3923),
    "pattaya": (12.9236, 100.8825),
    "khon_kaen": (16.4419, 102.8360),
    "hat_yai": (7.0086, 100.4747),
    "ayutthaya": (14.3532, 100.5689),
}


# ═══════════════════════════════════════════════════════════════
# Region profile
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RegionProfile:
    """Market-specific configuration."""
    code: str
    name: str
    country_code: str               # ISO-3166 alpha-2, for geocoder biasing
    currency: str                   # ISO-4217
    default_gsd_m: float            # m/pixel for hosted imagery layers
    # Default fallback prices (per m²) when price book unavailable
    price_per_m2: dict[str, float] = field(default_factory=dict)
    # Bounding box (min_lat, min_lng, max_lat, max_lng)
    bbox: tuple[float, float, float, float] | None = None
    # Hint for VLM prompt — roof material norms in this region
    roof_context_hint: str = ""
    # Preferred OSM locale for tags
    osm_locale: str = "en"


THAILAND_PROFILE = RegionProfile(
    code="TH",
    name="Thailand",
    country_code="th",
    currency="THB",
    default_gsd_m=0.3,
    price_per_m2={
        # THB prices tuned to the Thai market for corrugated zinc/metal roofs
        # Sources: approximate Thai supplier catalogs (BlueScope, SCG, etc.)
        # All prices include materials + labor.
        "replacement": 850.0,    # ~$24 USD/m² for full replacement
        "repair": 450.0,         # ~$13 USD/m² for patch repair
        "coating": 280.0,        # ~$8 USD/m² for anti-rust coating
        "inspection_flat": 2500.0,  # flat inspection fee
    },
    bbox=(
        THAILAND_BBOX["min_lat"], THAILAND_BBOX["min_lng"],
        THAILAND_BBOX["max_lat"], THAILAND_BBOX["max_lng"],
    ),
    roof_context_hint=(
        "This is a rooftop in Thailand. Common roof types here: "
        "corrugated galvanized zinc/steel (very common on residential, "
        "agricultural, and industrial buildings), painted colorbond steel, "
        "ceramic tile (traditional Thai and terracotta). The tropical "
        "monsoon climate (high humidity, heavy rain 6-9mo/yr, salt-laden "
        "coastal air) accelerates metal corrosion, so rust on older metal "
        "roofs is common and expected."
    ),
    osm_locale="th",
)


USA_PROFILE = RegionProfile(
    code="US",
    name="United States",
    country_code="us",
    currency="USD",
    default_gsd_m=0.3,
    price_per_m2={
        "replacement": 45.00,
        "repair": 25.00,
        "coating": 15.00,
        "inspection_flat": 150.00,
    },
    roof_context_hint="",
    osm_locale="en",
)


REGION_PROFILES: dict[str, RegionProfile] = {
    "TH": THAILAND_PROFILE,
    "US": USA_PROFILE,
}


# ═══════════════════════════════════════════════════════════════
# Active region selection
# ═══════════════════════════════════════════════════════════════

def get_active_region() -> RegionProfile:
    """Return the active region profile from the REGION env var (default: TH)."""
    code = os.environ.get("REGION", "TH").upper()
    return REGION_PROFILES.get(code, THAILAND_PROFILE)


def is_in_region(lat: float, lng: float, profile: RegionProfile | None = None) -> bool:
    """Check whether a coordinate falls within the active region's bbox."""
    profile = profile or get_active_region()
    if profile.bbox is None:
        return True  # global
    min_lat, min_lng, max_lat, max_lng = profile.bbox
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng
