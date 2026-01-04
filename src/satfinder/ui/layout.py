"""Main Gradio Blocks layout."""

import json
import gradio as gr

from ..config import DEFAULT_HEATMAP_OPACITY, DEFAULT_CITY, CITIES
from ..state import add_point, clear_points, compute_and_return_heatmap, get_city_config
from .controls import create_controls
from .viewer import create_viewer_html, get_message_handler_js


# CSS to hide state textboxes completely
# Target both the element and its parent container
# Use visibility: hidden instead of display: none to keep elements in DOM
CUSTOM_CSS = """
#click_xy, #points_json, #heatmap_data, #grid_toggle, #heatmap_toggle, #heatmap_opacity, #city_config,
#click_xy-container, #points_json-container, #heatmap_data-container,
#grid_toggle-container, #heatmap_toggle-container, #heatmap_opacity-container, #city_config-container {
    position: absolute !important;
    left: -9999px !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}
"""


def create_app() -> gr.Blocks:
    """Create the Gradio Blocks application.

    Returns:
        Configured Gradio Blocks instance.
    """
    with gr.Blocks(title="Satellite Structure Finder") as demo:
        # Inject CSS and JS via HTML component (works with mount_gradio_app)
        # visible=True required for CSS to be rendered in the page
        gr.HTML(
            f"<style>{CUSTOM_CSS}</style><script>({get_message_handler_js()})()</script>",
        )

        # Header
        gr.Markdown("""
# Satellite Structure Finder with DINOv3

**Select a city**, then **click on the map** to select example structures. Click **Find Similar** to highlight matching areas.
        """)

        with gr.Row():
            # Main viewer column
            with gr.Column(scale=4):
                # OpenSeadragon viewer
                gr.HTML(create_viewer_html())

                # Hidden state textboxes - visible=True so DOM elements exist, hidden via CSS
                click_xy = gr.Textbox(label="", elem_id="click_xy", visible=True)
                points_json = gr.Textbox(
                    label="", elem_id="points_json", value="[]", visible=True
                )
                heatmap_data = gr.Textbox(
                    label="", elem_id="heatmap_data", visible=True
                )
                grid_toggle = gr.Textbox(
                    label="", elem_id="grid_toggle", value="false", visible=True
                )
                heatmap_toggle = gr.Textbox(
                    label="", elem_id="heatmap_toggle", value="true", visible=True
                )
                heatmap_opacity = gr.Textbox(
                    label="",
                    elem_id="heatmap_opacity",
                    value=str(DEFAULT_HEATMAP_OPACITY),
                    visible=True,
                )
                # Hidden textbox for city configuration updates
                # Initialize with default city config
                default_city_cfg = CITIES[DEFAULT_CITY]
                initial_city_config = json.dumps(
                    {
                        "city": DEFAULT_CITY,
                        "IMG_W": default_city_cfg["img_w"],
                        "IMG_H": default_city_cfg["img_h"],
                        "GRID_W": default_city_cfg["grid_w"],
                        "GRID_H": default_city_cfg["grid_h"],
                        "DZI_URL": default_city_cfg["dzi_url"],
                    }
                )
                city_config = gr.Textbox(
                    label="",
                    elem_id="city_config",
                    value=initial_city_config,
                    visible=True,
                )

            # Sidebar controls
            controls = create_controls()

        # Wire up event handlers
        _wire_events(
            click_xy=click_xy,
            points_json=points_json,
            heatmap_data=heatmap_data,
            grid_toggle=grid_toggle,
            heatmap_toggle=heatmap_toggle,
            heatmap_opacity=heatmap_opacity,
            city_config=city_config,
            controls=controls,
        )

    return demo


def _wire_events(
    click_xy: gr.Textbox,
    points_json: gr.Textbox,
    heatmap_data: gr.Textbox,
    grid_toggle: gr.Textbox,
    heatmap_toggle: gr.Textbox,
    heatmap_opacity: gr.Textbox,
    city_config: gr.Textbox,
    controls: dict,
) -> None:
    """Wire up all event handlers.

    Args:
        click_xy: Hidden textbox receiving click coordinates
        points_json: Hidden textbox storing points state
        heatmap_data: Hidden textbox storing heatmap base64
        grid_toggle: Hidden textbox for grid visibility
        heatmap_toggle: Hidden textbox for heatmap visibility
        heatmap_opacity: Hidden textbox for opacity value
        city_config: Hidden textbox for city configuration updates
        controls: Dictionary of control components from create_controls()
    """
    # Click handler - add point when click_xy changes
    click_xy.change(
        fn=add_point,
        inputs=[
            click_xy,
            points_json,
            controls["point_type"],
            controls["city_selector"],
        ],
        outputs=[points_json, controls["log_output"]],
    )

    # Clear button
    controls["clear_btn"].click(
        fn=clear_points,
        inputs=[],
        outputs=[points_json, heatmap_data, controls["log_output"]],
    )

    # Find Similar button
    controls["find_btn"].click(
        fn=compute_and_return_heatmap,
        inputs=[points_json, controls["city_selector"]],
        outputs=[heatmap_data, controls["log_output"]],
    )

    # Download button - triggers JavaScript to capture canvas via postMessage
    controls["download_btn"].click(
        fn=None,
        inputs=[],
        outputs=[],
        js="""
        () => {
            const iframe = document.getElementById('osd-viewer-frame');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: 'download' }, '*');
            }
        }
        """,
    )

    # City selector change - update config and clear points
    def on_city_change(city: str):
        """Handle city change: return new config JSON and clear points."""
        config = get_city_config(city)
        config_json = json.dumps(
            {
                "city": city,
                "IMG_W": config["img_w"],
                "IMG_H": config["img_h"],
                "GRID_W": config["grid_w"],
                "GRID_H": config["grid_h"],
                "DZI_URL": config["dzi_url"],
            }
        )
        return config_json, "[]", "", f"Switched to {config['name']}"

    controls["city_selector"].change(
        fn=on_city_change,
        inputs=[controls["city_selector"]],
        outputs=[city_config, points_json, heatmap_data, controls["log_output"]],
    )

    # Grid toggle
    controls["grid_checkbox"].change(
        fn=lambda x: "true" if x else "false",
        inputs=[controls["grid_checkbox"]],
        outputs=[grid_toggle],
    )

    # Heatmap toggle
    controls["heatmap_checkbox"].change(
        fn=lambda x: "true" if x else "false",
        inputs=[controls["heatmap_checkbox"]],
        outputs=[heatmap_toggle],
    )

    # Opacity slider
    controls["opacity_slider"].change(
        fn=lambda x: str(x),
        inputs=[controls["opacity_slider"]],
        outputs=[heatmap_opacity],
    )
