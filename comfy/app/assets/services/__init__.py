"""Re-exports the asset service functions so routes and the seeder import them
from one place instead of reaching into individual service modules.
"""

from .asset_management import asset_exists
from .asset_management import delete_asset_reference
from .asset_management import get_asset_detail
from .asset_management import get_preview_file_paths
from .asset_management import resolve_asset_for_download
from .asset_management import update_asset_metadata
from .ingest import DependencyMissingError
from .ingest import HashMismatchError
from .ingest import UploadUnstableError
from .ingest import create_from_hash
from .ingest import register_file_in_place
from .ingest import upload_from_temp_path
from .tagging import apply_tags
from .tagging import list_tags
from .tagging import remove_tags

__all__ = [
    "DependencyMissingError",
    "HashMismatchError",
    "UploadUnstableError",
    "upload_from_temp_path",
    "create_from_hash",
    "register_file_in_place",
    "get_asset_detail",
    "update_asset_metadata",
    "delete_asset_reference",
    "asset_exists",
    "get_preview_file_paths",
    "resolve_asset_for_download",
    "apply_tags",
    "remove_tags",
    "list_tags",
]
