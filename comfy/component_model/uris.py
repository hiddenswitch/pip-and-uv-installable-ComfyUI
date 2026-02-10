def is_uri(path: str) -> bool:
    """Check if a path is a URI (has a scheme like https://, hf://, s3://, etc.)."""
    return "://" in path and not path.startswith("://")


def is_hf_uri(path: str) -> bool:
    """Check if a path is a Hugging Face URI (hf://)."""
    return path.startswith("hf://")
