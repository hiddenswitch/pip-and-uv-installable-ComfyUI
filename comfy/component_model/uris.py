from can_ada import can_parse, parse  # pylint: disable=no-name-in-module


def is_uri(path: str) -> bool:
    """Check if a path is a URI (has a scheme like https://, hf://, s3://, etc.)."""
    return can_parse(path)


def is_hf_uri(path: str) -> bool:
    """Check if a path is a Hugging Face URI (hf://)."""
    return can_parse(path) and parse(path).protocol == "hf:"
