"""Re-exports the record and tag query functions so callers import them from one
place instead of reaching into individual query modules. A module-level
``__getattr__`` resolves names that live in the tag module, keeping this import
surface flat as queries are split across more files.
"""

from importlib import import_module

from .records import create_content
from .records import create_content_reporting_insert
from .records import create_record
from .records import delete_record
from .records import fetch_record_tags
from .records import get_record_by_id
from .records import list_records_page
from .records import mark_content_missing
from .records import rename_record
from .records import unset_content_missing
from .records import update_record_access_time

__all__ = [
    "create_content",
    "create_content_reporting_insert",
    "create_record",
    "delete_record",
    "fetch_record_tags",
    "get_record_by_id",
    "list_records_page",
    "mark_content_missing",
    "rename_record",
    "unset_content_missing",
    "update_record_access_time",
]


def __getattr__(name: str):
    for module_name in ("tags",):
        module = import_module(f".{module_name}", __name__)
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(name)
