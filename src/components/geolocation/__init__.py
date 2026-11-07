"""
src/components/geolocation/__init__.py

Browser Geolocation component with safe fallback for VayuDrishti.
Adheres to modern browser security models and provides a safe fallback
when custom component iframes are restricted or disabled.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger("VayuDrishti.Geolocation")

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

# Configurable toggle: defaults to False to prevent iframe loading warnings,
# can be enabled by setting VAYU_ENABLE_GEO_COMPONENT=true
ENABLE_CUSTOM_COMPONENT = os.getenv("VAYU_ENABLE_GEO_COMPONENT", "false").strip().lower() in ("true", "1", "yes")

_component_func = None
if ENABLE_CUSTOM_COMPONENT:
    if _FRONTEND_DIR.exists() and (_FRONTEND_DIR / "index.html").exists():
        try:
            _component_func = components.declare_component(
                "vayu_geolocation",
                path=str(_FRONTEND_DIR),
            )
        except Exception as e:
            logger.warning("Could not declare vayu_geolocation custom component: %s", e)
            _component_func = None


def render_geolocation_button(
    key: str = "vayu_geo_button",
    button_label: str = "📍 Use My Current Location",
    is_detected: bool = False,
    default: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Renders an interactive 'Use My Current Location' button.

    If the custom component is disabled or unavailable, renders a native Streamlit button
    that clearly informs the user that automatic geolocation is temporarily disabled,
    preserving full manual city and station selection without any page-loading warnings.

    Returns:
        dict with geolocation result if available, or None.
    """
    if not ENABLE_CUSTOM_COMPONENT or _component_func is None:
        btn_text = "📍 Location Active" if is_detected else button_label
        if st.button(btn_text, key=f"{key}_fallback", use_container_width=True):
            st.info("ℹ️ Automatic browser geolocation is temporarily unavailable. Please select your location using the city search bar or sidebar.")
        return None

    try:
        return _component_func(
            button_label=button_label,
            is_detected=is_detected,
            key=key,
            default=default,
        )
    except Exception as e:
        logger.warning("Error rendering vayu_geolocation component: %s", e)
        return None
