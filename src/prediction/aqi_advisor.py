"""
src/prediction/aqi_advisor.py

Official Indian Central Pollution Control Board (CPCB) National Air Quality Index (NAQI)
sub-index calculation engine, category classifier, and public health risk advisor.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# Official CPCB Breakpoints Table (CPCB NAQI Standard, 2014 / NAAQS 2009)
# Format: (C_low, C_high, I_low, I_high)
BREAKPOINTS = {
    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "NO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500),
    ],
    "SO2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500),
    ],
    "CO": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "O3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ],
}

# Category metadata
AQI_CATEGORIES = [
    {
        "tier": "Good",
        "tier_te": "మంచిది",
        "min": 0,
        "max": 50,
        "color": "#16a34a",
        "bg_color": "#dcfce7",
        "health_impact": "Minimal impact. Air quality is considered satisfactory, and air pollution poses little or no risk.",
        "health_impact_te": "కనిష్ట ప్రభావం. గాలి నాణ్యత అనుకూలంగా ఉంది, ఆరోగ్యానికి ఎటువంటి ప్రమాదం లేదు.",
        "guidance": {
            "general": "Ideal for all outdoor activities and exercise.",
            "children": "Normal outdoor play and sports recommended.",
            "elderly": "Normal outdoor activities permitted.",
            "sensitive": "No special restrictions needed.",
        },
    },
    {
        "tier": "Satisfactory",
        "tier_te": "సంతృప్తికరం",
        "min": 51,
        "max": 100,
        "color": "#65a30d",
        "bg_color": "#ecfccb",
        "health_impact": "Minor breathing discomfort to sensitive people. Acceptable air quality for the general public.",
        "health_impact_te": "సున్నితమైన వ్యక్తులకు స్వల్ప శ్వాస ఇబ్బంది ఉండవచ్చు. సాధారణ ప్రజలకు ఆమోదయోగ్యమైనది.",
        "guidance": {
            "general": "Outdoor activities are generally safe.",
            "children": "Normal outdoor play allowed.",
            "elderly": "Can engage in normal activities; monitor for coughing if unusually sensitive.",
            "sensitive": "Individuals with acute asthma should keep quick-relief medication handy.",
        },
    },
    {
        "tier": "Moderate",
        "tier_te": "మధ్యస్థం",
        "min": 101,
        "max": 200,
        "color": "#ca8a04",
        "bg_color": "#fef9c3",
        "health_impact": "Breathing discomfort to people with lung disease (such as asthma) and discomfort to children and older adults.",
        "health_impact_te": "ఉబ్బసం ఉన్నవారికి, పిల్లలకు మరియు వృద్ధులకు శ్వాస తీసుకోవడంలో అసౌకర్యం కలగవచ్చు.",
        "guidance": {
            "general": "Take more breaks during prolonged outdoor exertion.",
            "children": "Reduce intense running games or sports lasting several hours outdoors.",
            "elderly": "Take regular rest periods during outdoor walks.",
            "sensitive": "Keep reliever inhalers nearby; avoid busy roads and heavy traffic zones.",
        },
    },
    {
        "tier": "Poor",
        "tier_te": "పేలవం",
        "min": 201,
        "max": 300,
        "color": "#ea580c",
        "bg_color": "#ffedd5",
        "health_impact": "Breathing discomfort to most people on prolonged exposure; significant discomfort for people with lung/heart disease.",
        "health_impact_te": "ఎక్కువ సేపు బయట ఉంటే చాలా మందికి శ్వాస తీసుకోవడంలో ఇబ్బంది కలుగుతుంది.",
        "guidance": {
            "general": "Cut back on prolonged or heavy outdoor exertion.",
            "children": "Limit outdoor playtime, especially during early morning and late evening rush hours.",
            "elderly": "Prefer indoor light activities; avoid strenuous walking in high pollution periods.",
            "sensitive": "Wear an N95 respirator mask if outdoor travel is necessary; consult a physician if wheezing occurs.",
        },
    },
    {
        "tier": "Very Poor",
        "tier_te": "చాలా పేలవం",
        "min": 301,
        "max": 400,
        "color": "#dc2626",
        "bg_color": "#fee2e2",
        "health_impact": "Respiratory illness on prolonged exposure. Effects may be more pronounced in people with lung and heart diseases.",
        "health_impact_te": "ఎక్కువ సేపు బయట గడిపితే శ్వాసకోశ వ్యాధులు వచ్చే ప్రమాదం ఉంది.",
        "guidance": {
            "general": "Avoid prolonged outdoor exertion; keep windows closed during smog peaks.",
            "children": "Avoid outdoor play; keep activities indoors.",
            "elderly": "Stay indoors as much as possible; keep indoor air clean.",
            "sensitive": "Strictly avoid outdoor activities; use HEPA air purifiers indoors if available; keep medications accessible.",
        },
    },
    {
        "tier": "Severe",
        "tier_te": "తీవ్రమైనది",
        "min": 401,
        "max": 500,
        "color": "#7f1d1d",
        "bg_color": "#fdf2f8",
        "health_impact": "Severe respiratory effects on healthy people and serious health impact on people with lung or heart disease.",
        "health_impact_te": "ఆరోగ్యవంతులకు కూడా తీవ్రమైన శ్వాసకోశ సమస్యలు వచ్చే అవకాశం ఉంది.",
        "guidance": {
            "general": "Avoid all physical outdoor exertion; remain indoors.",
            "children": "Schools and parents should keep children strictly indoors.",
            "elderly": "Remain indoors in well-sealed rooms; avoid any physical exertion.",
            "sensitive": "High health hazard; seek immediate medical consultation if experiencing chest pain, tightness, or shortness of breath.",
        },
    },
]


def calculate_sub_index(pollutant: str, concentration: float) -> Optional[int]:
    """Calculates the Indian CPCB sub-index for a specific pollutant concentration

    using piecewise linear interpolation over official breakpoint tables.
    """
    if pollutant not in BREAKPOINTS or concentration is None or np.isnan(concentration):
        return None

    conc = float(concentration)
    if conc < 0.0:
        return 0

    ranges = BREAKPOINTS[pollutant]
    for c_low, c_high, i_low, i_high in ranges:
        if c_low <= conc <= c_high:
            sub = i_low + ((i_high - i_low) / (c_high - c_low)) * (conc - c_low)
            return int(round(sub))

    # If concentration exceeds highest breakpoint, cap at 500
    if conc > ranges[-1][1]:
        return 500

    return None


def get_aqi_category(aqi_val: int) -> Dict[str, Any]:
    """Retrieves category metadata, health impacts, and guidance for a given AQI value."""
    aqi_clamped = max(0, min(500, int(round(aqi_val))))
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi_clamped <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]


def calculate_overall_cpcb_aqi(pollutants_dict: Dict[str, float]) -> Dict[str, Any]:
    """Calculates official CPCB National Air Quality Index (NAQI) following CPCB guidelines:

    - Requires at least 3 valid pollutants.
    - At least one pollutant must be particulate matter (PM2.5 or PM10).
    - Overall AQI is the maximum of the sub-indices (Prominent Pollutant).
    - If criteria are not met, returns 'AQI unavailable — insufficient data'.
    """
    sub_indices: Dict[str, int] = {}
    for p, val in pollutants_dict.items():
        if val is not None and not np.isnan(val):
            sub = calculate_sub_index(p, val)
            if sub is not None:
                sub_indices[p] = sub

    has_pm = ("PM2.5" in sub_indices) or ("PM10" in sub_indices)
    valid_count = len(sub_indices)

    # Official CPCB requirement check
    if valid_count >= 3 and has_pm:
        # Determine prominent pollutant (maximum sub-index)
        prominent = max(sub_indices.keys(), key=lambda k: sub_indices[k])
        overall_aqi = sub_indices[prominent]
        cat = get_aqi_category(overall_aqi)

        return {
            "aqi": overall_aqi,
            "status": "Official CPCB NAQI Valid",
            "is_valid": True,
            "prominent_pollutant": prominent,
            "sub_indices": sub_indices,
            "category": cat["tier"],
            "category_te": cat["tier_te"],
            "color": cat["color"],
            "bg_color": cat["bg_color"],
            "health_impact": cat["health_impact"],
            "health_impact_te": cat["health_impact_te"],
            "guidance": cat["guidance"],
            "notes": f"Calculated across {valid_count} monitored pollutants with prominent pollutant {prominent}.",
        }
    else:
        # Insufficient data for official composite AQI
        reasons = []
        if valid_count < 3:
            reasons.append(f"Only {valid_count} pollutant(s) available (minimum 3 required per CPCB)")
        if not has_pm:
            reasons.append("Neither PM2.5 nor PM10 present (mandatory particulate requirement)")

        # Provide individual sub-indices if any exist
        primary_sub = None
        if "PM2.5" in sub_indices:
            primary_sub = {"pollutant": "PM2.5", "sub_index": sub_indices["PM2.5"]}
        elif "PM10" in sub_indices:
            primary_sub = {"pollutant": "PM10", "sub_index": sub_indices["PM10"]}

        cat = get_aqi_category(primary_sub["sub_index"]) if primary_sub else AQI_CATEGORIES[2]

        return {
            "aqi": primary_sub["sub_index"] if primary_sub else None,
            "status": "AQI unavailable — insufficient data",
            "is_valid": False,
            "prominent_pollutant": primary_sub["pollutant"] if primary_sub else "None",
            "sub_indices": sub_indices,
            "category": f"{primary_sub['pollutant']} Sub-Index: {cat['tier']}" if primary_sub else "Unavailable",
            "category_te": f"{primary_sub['pollutant']} ఉప-సూచిక: {cat['tier_te']}" if primary_sub else "అందుబాటులో లేదు",
            "color": cat["color"] if primary_sub else "#64748b",
            "bg_color": cat["bg_color"] if primary_sub else "#f1f5f9",
            "health_impact": cat["health_impact"] if primary_sub else "Insufficient pollutant data to determine health risk.",
            "health_impact_te": cat["health_impact_te"] if primary_sub else "ఆరోగ్య ప్రమాదాన్ని అంచనా వేయడానికి తగినంత కాలుష్య కారక సమాచారం లేదు.",
            "guidance": cat["guidance"] if primary_sub else {},
            "notes": "Official composite CPCB AQI requires at least 3 pollutants including PM2.5/PM10. " + "; ".join(reasons) + ".",
        }
