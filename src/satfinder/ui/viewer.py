"""OpenSeadragon viewer integration."""

import json

from ..config import (
    CITIES,
    DEFAULT_CITY,
    MIN_GRID_SPACING_PX,
    GRID_OPACITY,
    DEFAULT_HEATMAP_OPACITY,
    VIEWER_HEIGHT_PX,
)


def get_config_json(city: str = DEFAULT_CITY) -> str:
    """Get configuration as JSON for injection into JavaScript.

    Args:
        city: City key to get configuration for

    Returns:
        JSON string with all configuration values.
    """
    city_config = CITIES.get(city, CITIES[DEFAULT_CITY])
    config = {
        "IMG_W": city_config["img_w"],
        "IMG_H": city_config["img_h"],
        "GRID_W": city_config["grid_w"],
        "GRID_H": city_config["grid_h"],
        "MIN_GRID_SPACING": MIN_GRID_SPACING_PX,
        "GRID_OPACITY": GRID_OPACITY,
        "DEFAULT_HEATMAP_OPACITY": DEFAULT_HEATMAP_OPACITY,
        "DZI_URL": city_config["dzi_url"],
    }
    return json.dumps(config)


def get_all_cities_config() -> str:
    """Get configuration for all cities as JSON.

    Returns:
        JSON string with city configurations.
    """
    cities_config = {}
    for key, city in CITIES.items():
        cities_config[key] = {
            "name": city["name"],
            "IMG_W": city["img_w"],
            "IMG_H": city["img_h"],
            "GRID_W": city["grid_w"],
            "GRID_H": city["grid_h"],
            "DZI_URL": city["dzi_url"],
        }
    return json.dumps(cities_config)


def create_viewer_html() -> str:
    """Generate the OpenSeadragon viewer HTML using an iframe.

    The iframe loads viewer.html which handles the map display.
    Configuration is passed via a global variable.

    Returns:
        HTML string for embedding in Gradio.
    """
    config_json = get_config_json(DEFAULT_CITY)
    cities_json = get_all_cities_config()

    return f"""
<script>
    // Inject configuration for viewer.html to read
    window.SATFINDER_CONFIG = {config_json};
    window.SATFINDER_CITIES = {cities_json};
    window.SATFINDER_CURRENT_CITY = "{DEFAULT_CITY}";
</script>
<iframe
    id="osd-viewer-frame"
    src="/static/viewer.html"
    style="width: 100%; height: {VIEWER_HEIGHT_PX}px; border: none; border-radius: 8px;"
></iframe>
"""


def get_message_handler_js() -> str:
    """Get JavaScript for handling messages from the iframe.

    This script receives click coordinates from the iframe and
    updates Gradio's hidden textbox to trigger Python handlers.

    Returns:
        JavaScript code as string.
    """
    return """
function() {
    // Set up message listener for iframe communication
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'click') {
            console.log('[gradio] Received click:', event.data.x, event.data.y);
            const newValue = event.data.x.toFixed(1) + ',' + event.data.y.toFixed(1);

            // Find the click_xy textbox
            const clickBox = document.querySelector('#click_xy input, #click_xy textarea');
            if (clickBox) {
                // Use native input value setter and dispatch events
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    clickBox.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
                    'value'
                ).set;
                nativeSetter.call(clickBox, newValue);

                // Dispatch input event to trigger Gradio's change handler
                clickBox.dispatchEvent(new Event('input', { bubbles: true }));
                console.log('[gradio] Set click_xy to:', newValue);
            }
        }
    });
    console.log('[gradio] Message listener initialized');
}
"""
