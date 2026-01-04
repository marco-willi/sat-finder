"""Sidebar control components."""

import gradio as gr

from ..config import DEFAULT_HEATMAP_OPACITY, CITIES, DEFAULT_CITY


def create_controls() -> dict:
    """Create the sidebar controls section.

    Returns:
        Dictionary containing all control components for event wiring.
    """
    with gr.Column(scale=1) as container:
        gr.Markdown("### City Selection")
        city_choices = [(config["name"], key) for key, config in CITIES.items()]
        city_selector = gr.Dropdown(
            choices=city_choices,
            value=DEFAULT_CITY,
            label="Select City",
            interactive=True,
        )

        gr.Markdown("### Controls")

        find_btn = gr.Button("Find Similar", variant="primary", size="lg")
        clear_btn = gr.Button("Clear Points", variant="secondary")
        download_btn = gr.Button("📷 Download View", variant="secondary")

        gr.Markdown("### Point Type")
        point_type = gr.Radio(
            choices=[("Positive (+)", "pos"), ("Negative (-)", "neg")],
            value="pos",
            label="Click adds:",
            interactive=True,
        )

        gr.Markdown("### Display Options")
        grid_checkbox = gr.Checkbox(label="Show DINO Grid", value=False)
        heatmap_checkbox = gr.Checkbox(label="Show Heatmap", value=True)
        opacity_slider = gr.Slider(
            label="Heatmap Opacity",
            minimum=0.0,
            maximum=1.0,
            value=DEFAULT_HEATMAP_OPACITY,
            step=0.05,
        )

        gr.Markdown("### Log")
        log_output = gr.Textbox(show_label=False, lines=4, interactive=False)

        gr.Markdown("""
---
### Instructions
1. **Select a city** from the dropdown
2. Select point type (positive/negative)
3. **Click** on structures you want to find
4. Click **Find Similar** to compute similarity
5. Similar areas will be highlighted in red

**Positive points**: Examples of what to find
**Negative points**: Examples of what to avoid
        """)

    return {
        "container": container,
        "city_selector": city_selector,
        "find_btn": find_btn,
        "clear_btn": clear_btn,
        "download_btn": download_btn,
        "point_type": point_type,
        "grid_checkbox": grid_checkbox,
        "heatmap_checkbox": heatmap_checkbox,
        "opacity_slider": opacity_slider,
        "log_output": log_output,
    }
