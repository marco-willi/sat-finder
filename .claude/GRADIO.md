# Gradio Best Practices and hf_app Structure

## Current hf_app Structure (Refactored)

### File Organization
```
hf_app/
├── app.py                        # Entry point (44 lines)
├── requirements.txt
├── packages.txt
│
├── satfinder_gradio/             # Main package (779 lines total)
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Centralized configuration (60 lines)
│   ├── similarity.py             # Business logic (113 lines)
│   ├── visualization.py          # Heatmap generation (102 lines)
│   ├── state.py                  # Event handlers (119 lines)
│   └── ui/
│       ├── __init__.py
│       ├── layout.py             # Main Blocks layout (143 lines)
│       ├── controls.py           # Sidebar controls (63 lines)
│       └── viewer.py             # OpenSeadragon integration (96 lines)
│
├── assets/
│   ├── embeddings.npz            # Pre-computed DINOv3 features (599MB)
│   └── map.jpg
│
└── static/
    ├── viewer.html               # OpenSeadragon viewer (reads config from parent)
    ├── js/openseadragon.min.js
    └── tiles/scene.dzi + scene_files/
```

### Architecture Improvements

| Before | After | Benefit |
|--------|-------|---------|
| Monolithic 698-line app.py | 10 focused modules | Maintainable, testable |
| Duplicated config | Single `config.py` | Single source of truth |
| JS in Python f-strings | Config injected via `window.SATFINDER_CONFIG` | Clean separation |
| Mixed concerns | Separated business logic, UI, state | Easy to extend |

---

## Gradio Best Practices (from official docs)

### 1. Code Organization

**Use `gr.Blocks()` for complex apps** - Enables custom layouts and multiple data flows.

**Separate concerns:**
- Configuration in dedicated module
- Business logic separate from UI
- State management isolated
- UI components modular

### 2. State Management Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Global State** | Shared across users (models, embeddings) | Module-level variables |
| **Session State** | Per-user, persists across interactions | `gr.State()` |
| **Browser State** | Persists across refreshes | `gr.BrowserState()` |

### 3. Custom CSS/JS Integration

```python
# Preferred: Use elem_id for stable targeting
component = gr.Component(elem_id="my-component")

# CSS via parameter (not inline strings)
with gr.Blocks(css_paths=["style.css"]) as demo:
    ...

# JS for initial setup only
with gr.Blocks(js="path/to/init.js") as demo:
    ...
```

### 4. Event Handling

```python
# Chain related events
button.click(fn=step1, ...).then(fn=step2, ...)

# Multiple triggers, single handler
gr.on(
    triggers=[btn.click, textbox.submit],
    fn=handler,
    inputs=[...],
    outputs=[...]
)
```

---

## Recommended Refactored Structure

```
hf_app/
├── app.py                    # Entry point only (~50 lines)
├── requirements.txt
├── packages.txt
│
├── satfinder_gradio/         # Package for Gradio-specific code
│   ├── __init__.py
│   ├── config.py             # All configuration constants
│   ├── state.py              # State management (gr.State wrappers)
│   ├── similarity.py         # Business logic (compute_similarity, etc.)
│   ├── visualization.py      # Heatmap generation
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── layout.py         # Main Blocks layout
│   │   ├── controls.py       # Sidebar controls component
│   │   └── viewer.py         # OpenSeadragon integration
│   └── assets/               # Static assets
│       └── ...
│
├── static/
│   ├── css/
│   │   └── app.css           # External CSS (not inline)
│   ├── js/
│   │   ├── openseadragon.min.js
│   │   ├── viewer-init.js    # Viewer initialization
│   │   └── gradio-bridge.js  # Gradio communication
│   └── tiles/
│       └── ...
│
└── tests/
    ├── test_similarity.py
    └── test_visualization.py
```

---

## Specific Refactoring Recommendations

### 1. Extract Configuration (`satfinder_gradio/config.py`)

```python
"""Configuration constants shared across modules."""
from pathlib import Path

# Image dimensions
IMG_W = 5888
IMG_H = 7168

# Embedding grid
GRID_W = 368
GRID_H = 448
PATCH_SIZE = 16

# Paths
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"
STATIC_DIR = BASE_DIR / "static"

# URLs
DZI_URL = "/static/tiles/scene.dzi"
OSD_JS_URL = "/static/js/openseadragon.min.js"

# Display settings
MIN_GRID_SPACING_PX = 12
GRID_OPACITY = 0.25
DEFAULT_HEATMAP_OPACITY = 0.6
```

### 2. Extract Business Logic (`satfinder_gradio/similarity.py`)

```python
"""Similarity computation - no Gradio dependencies."""
import numpy as np
from functools import lru_cache
from .config import ASSETS_DIR, GRID_W, GRID_H, IMG_W, IMG_H

@lru_cache(maxsize=1)
def load_embeddings() -> np.ndarray:
    """Load pre-computed embeddings (cached)."""
    emb_path = ASSETS_DIR / "embeddings.npz"
    if emb_path.exists():
        return np.load(emb_path)["features"]
    return np.zeros((GRID_H, GRID_W, 1024), dtype=np.float32)

def pixel_to_token(x: float, y: float) -> tuple[int, int]:
    """Convert pixel coordinates to token grid indices."""
    token_x = int(x / (IMG_W / GRID_W))
    token_y = int(y / (IMG_H / GRID_H))
    return (
        max(0, min(GRID_W - 1, token_x)),
        max(0, min(GRID_H - 1, token_y))
    )

def compute_similarity(points: list[dict]) -> np.ndarray:
    """Compute similarity map from positive/negative points."""
    # ... (existing logic, unchanged)
```

### 3. Use Proper Gradio State (`satfinder_gradio/state.py`)

```python
"""State management using gr.State instead of hidden textboxes."""
import gradio as gr
from dataclasses import dataclass, field

@dataclass
class AppState:
    """Application state container."""
    points: list[dict] = field(default_factory=list)
    heatmap_base64: str = ""
    show_grid: bool = True
    show_heatmap: bool = True
    heatmap_opacity: float = 0.6

def create_state() -> gr.State:
    """Create a new Gradio state with defaults."""
    return gr.State(AppState())

def add_point(state: AppState, x: float, y: float, label: str = "pos") -> AppState:
    """Add a point to state (immutable update)."""
    new_points = state.points + [{"x": x, "y": y, "label": label}]
    return AppState(
        points=new_points,
        heatmap_base64=state.heatmap_base64,
        show_grid=state.show_grid,
        show_heatmap=state.show_heatmap,
        heatmap_opacity=state.heatmap_opacity,
    )
```

### 4. External JavaScript Files

Move JS from Python f-strings to external files:

**`static/js/viewer-init.js`:**
```javascript
// Configuration injected via data attributes or global
const CONFIG = window.SATFINDER_CONFIG || {};

function initOpenSeadragon() {
    // ... clean JS without Python f-string escaping
}
```

**`static/js/gradio-bridge.js`:**
```javascript
// Handle Gradio <-> iframe communication
class GradioBridge {
    constructor(iframe) {
        this.iframe = iframe;
        this.setupListeners();
    }

    setupListeners() {
        window.addEventListener('message', this.handleMessage.bind(this));
    }

    handleMessage(event) {
        // ... message handling
    }
}
```

### 5. Modular UI Components (`satfinder_gradio/ui/controls.py`)

```python
"""Sidebar control components."""
import gradio as gr

def create_controls():
    """Create the sidebar controls section."""
    with gr.Column(scale=1) as controls:
        gr.Markdown("### Controls")

        find_btn = gr.Button("Find Similar", variant="primary", size="lg")
        clear_btn = gr.Button("Clear Points", variant="secondary")

        gr.Markdown("### Display Options")
        grid_checkbox = gr.Checkbox(label="Show DINO Grid", value=True)
        heatmap_checkbox = gr.Checkbox(label="Show Heatmap", value=True)
        opacity_slider = gr.Slider(
            label="Heatmap Opacity",
            minimum=0.0, maximum=1.0, value=0.6, step=0.05
        )

        gr.Markdown("### Log")
        log_output = gr.Textbox(label="", lines=4, interactive=False)

    return {
        "container": controls,
        "find_btn": find_btn,
        "clear_btn": clear_btn,
        "grid_checkbox": grid_checkbox,
        "heatmap_checkbox": heatmap_checkbox,
        "opacity_slider": opacity_slider,
        "log_output": log_output,
    }
```

### 6. Clean Entry Point (`app.py`)

```python
"""Satellite Structure Finder - Gradio Application."""
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from satfinder_gradio.config import STATIC_DIR
from satfinder_gradio.ui.layout import create_app
from satfinder_gradio.similarity import load_embeddings

# Pre-load embeddings
load_embeddings()

# Create FastAPI app with static files
fastapi_app = FastAPI()
fastapi_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Create and mount Gradio app
demo = create_app()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

---

## Alternative: Gradio Custom Components

For complex integrations like OpenSeadragon, consider creating a **Gradio Custom Component**:

```bash
gradio cc create OpenSeadragonViewer
```

This provides:
- Type-safe Python <-> JS communication
- Proper event handling (no polling)
- Reusable across projects
- Better testing support

See: https://www.gradio.app/guides/custom-components-in-five-minutes

---

## Migration Status (Completed)

| Task | Status |
|------|--------|
| Extract config.py | ✅ Done |
| Extract similarity.py (business logic) | ✅ Done |
| Extract visualization.py | ✅ Done |
| Create state.py (event handlers) | ✅ Done |
| Modularize UI components | ✅ Done |
| Update viewer.html to read config from parent | ✅ Done |
| Minimal app.py entry point | ✅ Done |

### Future Improvements

| Task | Effort | Impact |
|------|--------|--------|
| Create custom Gradio component for OpenSeadragon | High | High |
| Replace polling with proper events | Medium | Medium |
| Add unit tests for business logic | Low | Medium |

---

## References

- [Gradio Blocks Guide](https://www.gradio.app/guides/blocks-and-event-listeners)
- [Gradio State Management](https://www.gradio.app/guides/state-in-blocks)
- [Gradio Custom CSS/JS](https://www.gradio.app/guides/custom-CSS-and-JS)
- [Gradio Custom Components](https://www.gradio.app/guides/custom-components-in-five-minutes)
- [Gradio Layout Guide](https://www.gradio.app/guides/controlling-layout)
