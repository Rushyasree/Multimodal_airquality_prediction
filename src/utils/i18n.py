"""
src/utils/i18n.py

Bilingual localization engine providing English and Telugu (తెలుగు)
strings, labels, guidance, and scientific disclaimers for VayuDrishti.
"""

from typing import Dict, Any

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "en": "VayuDrishti",
        "te": "వాయుదృష్టి",
    },
    "app_tagline": {
        "en": "Air Quality Intelligence for India",
        "te": "భారతదేశానికి వాయు నాణ్యత సమాచారం",
    },
    "hero_desc": {
        "en": "An empirical deep learning system for high-resolution particulate matter (PM2.5 & PM10) forecasting across Indian urban environments, featuring an Andhra Pradesh location explorer, official CPCB health-danger advisory, and computer vision.",
        "te": "భారతీయ నగరాలలో PM2.5 మరియు PM10 కాలుష్య కారకాలను అంచనా వేసే డీప్ లెర్నింగ్ వ్యవస్థ. ఆంధ్రప్రదేశ్ లొకేషన్ ఎక్స్‌ప్లోరర్, అధికారిక CPCB ఆరోగ్య సలహాదారు మరియు కంప్యూటర్ విజన్ సౌకర్యాలతో రూపొందించబడింది.",
    },
    "nav_home": {
        "en": "🏠 Home & Overview",
        "te": "🏠 హోమ్ & అవలోకనం",
    },
    "nav_advisor": {
        "en": "🛡️ Smart Air Danger Advisor",
        "te": "🛡️ స్మార్ట్ ఎయిర్ డేంజర్ అడ్వైజర్",
    },
    "nav_map": {
        "en": "🗺️ Location Map & AP Explorer",
        "te": "🗺️ ఆంధ్రప్రదేశ్ & జాతీయ మ్యాప్",
    },
    "nav_forecast": {
        "en": "📈 Air-Quality Forecasting",
        "te": "📈 వాయు నాణ్యత సూచన (ఫోర్‌కాస్ట్)",
    },
    "nav_vision": {
        "en": "📷 Environmental Vision",
        "te": "📷 పర్యావరణ చిత్ర విశ్లేషణ",
    },
    "nav_history": {
        "en": "📊 Historical Analytics",
        "te": "📊 చారిత్రక విశ్లేషణ",
    },
    "nav_eval": {
        "en": "🎯 Model Evaluation & Benchmarks",
        "te": "🎯 మోడల్ మూల్యాంకనం & ఫలితాలు",
    },
    "nav_info": {
        "en": "ℹ️ System Health & Data Schema",
        "te": "ℹ️ సిస్టమ్ సమాచారం & స్కీమా",
    },
    "how_safe": {
        "en": "How safe is your air?",
        "te": "మీ గాలి ఎంత సురక్షితం?",
    },
    "what_to_do": {
        "en": "What should I do?",
        "te": "నేను ఏమి చేయాలి?",
    },
    "target_time": {
        "en": "Target Forecast Timestamp (t+1)",
        "te": "అంచనా సమయం (t+1)",
    },
    "predicted_pm25": {
        "en": "Predicted PM2.5 (t+1)",
        "te": "అంచనా వేసిన PM2.5 (t+1)",
    },
    "predicted_pm10": {
        "en": "Predicted PM10 (t+1)",
        "te": "అంచనా వేసిన PM10 (t+1)",
    },
    "persistence": {
        "en": "Persistence Benchmark (t)",
        "te": "మునుపటి వాస్తవ రీడింగ్ (t)",
    },
    "true_ground_truth": {
        "en": "True Ground Truth",
        "te": "నిజమైన వాస్తవ విలువ",
    },
    "proxy_forecast": {
        "en": "Nearby-Station Proxy Forecast",
        "te": "సమీప స్టేషన్ ప్రాక్సీ అంచనా",
    },
    "data_unavailable": {
        "en": "Historical model data unavailable",
        "te": "చారిత్రక మోడల్ డేటా అందుబాటులో లేదు",
    },
    "medical_disclaimer": {
        "en": "DISCLAIMER: This guidance represents general public health advice based on CPCB categories and does NOT constitute an individualized medical diagnosis or prescription.",
        "te": "గమనిక: ఈ మార్గదర్శకాలు CPCB వర్గాల ఆధారంగా సాధారణ ప్రజారోగ్య సలహాను మాత్రమే సూచిస్తాయి. ఇది వ్యక్తిగత వైద్య నిర్ధారణ లేదా చికిత్స కాదు.",
    },
    "select_state": {
        "en": "Select State / Region:",
        "te": "రాష్ట్రాన్ని ఎంచుకోండి:",
    },
    "select_city": {
        "en": "Select City / District:",
        "te": "నగరాన్ని / జిల్లాను ఎంచుకోండి:",
    },
    "select_station": {
        "en": "Select Monitoring Station / Location:",
        "te": "పర్యవేక్షణ స్టేషన్ / ప్రాంతాన్ని ఎంచుకోండి:",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Retrieves localized text string with fallback to English."""
    lang_key = "te" if lang == "te" else "en"
    if key in TRANSLATIONS and lang_key in TRANSLATIONS[key]:
        return TRANSLATIONS[key][lang_key]
    return TRANSLATIONS.get(key, {}).get("en", key)
