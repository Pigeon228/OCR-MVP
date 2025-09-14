import base64
import io
from PIL import Image

try:  # optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None


def pil_to_data_url(img: Image.Image) -> str:
    """Convert a PIL image to a data URL."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def np_convert(obj):
    """Convert NumPy types for JSON serialization."""
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    return str(obj)
