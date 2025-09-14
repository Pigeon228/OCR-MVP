def process_image(*args, **kwargs):
    from .pipeline import process_image as _process_image

    return _process_image(*args, **kwargs)


__all__ = ["process_image"]
