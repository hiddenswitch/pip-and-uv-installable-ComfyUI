from .asset import (
    asset_exists_by_hash,
    bulk_insert_assets,
    get_asset_by_hash,
    get_existing_asset_ids,
    reassign_asset_references,
    update_asset_hash_and_mime,
    upsert_asset,
)
from .asset_reference import (
    CacheStateRow,
    UnenrichedReferenceRow,
    bulk_insert_references_ignore_conflicts,
    bulk_update_enrichment_level,
    bulk_update_is_missing,
    bulk_update_needs_verify,
    convert_metadata_to_rows,
    delete_assets_by_ids,
    delete_orphaned_seed_asset,
    delete_reference_by_id,
    delete_references_by_ids,
    fetch_reference_and_asset,
    fetch_reference_asset_and_tags,
    get_or_create_reference,
    get_reference_by_file_path,
    get_reference_by_id,
    get_reference_with_owner_check,
    get_reference_ids_by_ids,
    get_references_by_paths_and_asset_ids,
    get_references_for_prefixes,
    get_unenriched_references,
    get_unreferenced_unhashed_asset_ids,
    insert_reference,
    list_references_by_asset_id,
    list_references_page,
    mark_references_missing_outside_prefixes,
    reference_exists_for_asset_id,
    restore_references_by_paths,
    set_reference_metadata,
    set_reference_preview,
    soft_delete_reference_by_id,
    update_reference_access_time,
    update_reference_name,
    update_reference_timestamps,
    update_reference_updated_at,
    upsert_reference,
)
from .tags import (
    AddTagsResult,
    RemoveTagsResult,
    SetTagsResult,
    add_missing_tag_for_asset_id,
    add_tags_to_reference,
    bulk_insert_tags_and_meta,
    ensure_tags_exist,
    get_reference_tags,
    list_tags_with_usage,
    remove_missing_tag_for_asset_id,
    remove_tags_from_reference,
    set_reference_tags,
    validate_tags_exist,
)
from ...helpers import select_best_live_path as pick_best_live_path


def asset_info_exists_for_asset_id(session, asset_id: str) -> bool:
    return reference_exists_for_asset_id(session, asset_id=asset_id)


def get_asset_info_by_id(session, asset_info_id: str):
    return get_reference_by_id(session, reference_id=asset_info_id)


def fetch_asset_info_asset_and_tags(session, asset_info_id: str, owner_id: str = ""):
    return fetch_reference_asset_and_tags(
        session, reference_id=asset_info_id, owner_id=owner_id
    )


def fetch_asset_info_and_asset(session, asset_info_id: str, owner_id: str = ""):
    return fetch_reference_and_asset(session, reference_id=asset_info_id, owner_id=owner_id)


def create_asset_info_for_existing_asset(
    session,
    asset_hash: str,
    name: str,
    user_metadata: dict | None = None,
    tags: list[str] | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
):
    asset = get_asset_by_hash(session, asset_hash=asset_hash)
    if not asset:
        raise ValueError(f"No asset with hash {asset_hash}")

    ref, _ = get_or_create_reference(
        session,
        asset_id=asset.id,
        owner_id=owner_id,
        name=name,
    )
    if user_metadata:
        set_reference_metadata(session, reference_id=ref.id, user_metadata=user_metadata)
    if tags is not None:
        set_reference_tags(session, reference_id=ref.id, tags=tags, origin=tag_origin)
    session.flush()
    return get_reference_by_id(session, reference_id=ref.id)


def touch_asset_info_by_id(session, asset_info_id: str):
    update_reference_access_time(session, reference_id=asset_info_id)


def update_asset_info_full(
    session,
    asset_info_id: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    tag_origin: str = "manual",
    asset_info_row=None,
):
    ref = asset_info_row or get_reference_by_id(session, reference_id=asset_info_id)
    if ref is None:
        raise ValueError(f"AssetReference {asset_info_id} not found")
    if name is not None:
        update_reference_name(session, reference_id=asset_info_id, name=name)
    if user_metadata is not None:
        set_reference_metadata(
            session, reference_id=asset_info_id, user_metadata=user_metadata
        )
    if tags is not None:
        set_reference_tags(
            session,
            reference_id=asset_info_id,
            tags=tags,
            origin=tag_origin,
        )
    session.flush()
    return get_reference_by_id(session, reference_id=asset_info_id)


def delete_asset_info_by_id(session, asset_info_id: str, owner_id: str = "") -> bool:
    return delete_reference_by_id(
        session, reference_id=asset_info_id, owner_id=owner_id
    )


def list_cache_states_by_asset_id(session, asset_id: str):
    return list_references_by_asset_id(session, asset_id=asset_id)


def list_asset_infos_page(
    session,
    owner_id: str = "",
    include_tags=None,
    exclude_tags=None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
):
    return list_references_page(
        session,
        owner_id=owner_id,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
        name_contains=name_contains,
        metadata_filter=metadata_filter,
        limit=limit,
        offset=offset,
        sort=sort,
        order=order,
    )


def get_asset_tags(session, asset_info_id: str) -> list[str]:
    return get_reference_tags(session, reference_id=asset_info_id)


def add_tags_to_asset_info(
    session,
    asset_info_id: str,
    tags: list[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    asset_info_row=None,
):
    result = add_tags_to_reference(
        session,
        reference_id=asset_info_id,
        tags=tags,
        origin=origin,
        create_if_missing=create_if_missing,
        reference_row=asset_info_row,
    )
    return {
        "added": result.added,
        "already_present": result.already_present,
        "total_tags": result.total_tags,
    }


def remove_tags_from_asset_info(session, asset_info_id: str, tags: list[str]):
    result = remove_tags_from_reference(session, reference_id=asset_info_id, tags=tags)
    return {
        "removed": result.removed,
        "not_present": result.not_present,
        "total_tags": result.total_tags,
    }


def ingest_fs_asset(
    session,
    asset_hash: str,
    abs_path: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    owner_id: str = "",
    preview_id: str | None = None,
    user_metadata: dict | None = None,
    tags=None,
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
):
    asset, asset_created, _ = upsert_asset(
        session,
        asset_hash=asset_hash,
        size_bytes=size_bytes,
        mime_type=mime_type,
    )
    _, _ = upsert_reference(
        session,
        asset_id=asset.id,
        file_path=abs_path,
        name=info_name or abs_path,
        mtime_ns=mtime_ns,
        owner_id=owner_id,
    )
    ref = get_reference_by_file_path(session, file_path=abs_path)
    if ref is None:
        raise RuntimeError("AssetReference row not found after ingest")
    if preview_id is not None:
        set_reference_preview(
            session, reference_id=ref.id, preview_asset_id=preview_id
        )
    if user_metadata:
        set_reference_metadata(session, reference_id=ref.id, user_metadata=user_metadata)
    if tags is not None:
        if require_existing_tags:
            validate_tags_exist(session, tags)
        add_tags_to_reference(
            session,
            reference_id=ref.id,
            tags=tags,
            origin=tag_origin,
            create_if_missing=not require_existing_tags,
        )
    session.flush()
    return {"asset_info_id": ref.id, "asset_created": asset_created}


def set_asset_info_preview(
    session, asset_info_id: str, preview_asset_id: str | None = None
):
    set_reference_preview(
        session, reference_id=asset_info_id, preview_asset_id=preview_asset_id
    )

__all__ = [
    "AddTagsResult",
    "CacheStateRow",
    "RemoveTagsResult",
    "SetTagsResult",
    "UnenrichedReferenceRow",
    "add_missing_tag_for_asset_id",
    "add_tags_to_reference",
    "asset_exists_by_hash",
    "asset_info_exists_for_asset_id",
    "bulk_insert_assets",
    "create_asset_info_for_existing_asset",
    "bulk_insert_references_ignore_conflicts",
    "bulk_insert_tags_and_meta",
    "bulk_update_enrichment_level",
    "bulk_update_is_missing",
    "bulk_update_needs_verify",
    "convert_metadata_to_rows",
    "delete_asset_info_by_id",
    "delete_assets_by_ids",
    "delete_orphaned_seed_asset",
    "delete_reference_by_id",
    "delete_references_by_ids",
    "ensure_tags_exist",
    "fetch_reference_and_asset",
    "fetch_reference_asset_and_tags",
    "fetch_asset_info_and_asset",
    "fetch_asset_info_asset_and_tags",
    "get_asset_by_hash",
    "get_asset_info_by_id",
    "get_existing_asset_ids",
    "get_or_create_reference",
    "get_reference_by_file_path",
    "get_reference_by_id",
    "get_reference_with_owner_check",
    "get_reference_ids_by_ids",
    "get_reference_tags",
    "get_asset_tags",
    "get_references_by_paths_and_asset_ids",
    "get_references_for_prefixes",
    "get_unenriched_references",
    "get_unreferenced_unhashed_asset_ids",
    "ingest_fs_asset",
    "insert_reference",
    "list_asset_infos_page",
    "list_cache_states_by_asset_id",
    "list_references_by_asset_id",
    "list_references_page",
    "list_tags_with_usage",
    "mark_references_missing_outside_prefixes",
    "pick_best_live_path",
    "reassign_asset_references",
    "reference_exists_for_asset_id",
    "remove_missing_tag_for_asset_id",
    "remove_tags_from_reference",
    "remove_tags_from_asset_info",
    "restore_references_by_paths",
    "set_asset_info_preview",
    "set_reference_metadata",
    "set_reference_preview",
    "soft_delete_reference_by_id",
    "touch_asset_info_by_id",
    "update_asset_info_full",
    "set_reference_tags",
    "update_asset_hash_and_mime",
    "update_reference_access_time",
    "update_reference_name",
    "update_reference_timestamps",
    "update_reference_updated_at",
    "upsert_asset",
    "upsert_reference",
    "validate_tags_exist",
]
