from sqlalchemy.orm import Session
import sqlalchemy as sa

from comfy.app.assets.database.models import AssetContent
from comfy.app.assets.database.queries import create_content
from comfy.app.assets.database.queries import create_record


def test_hash_is_a_content_fact(session: Session) -> None:
    content = create_content(session, "/models/a.safetensors", hash="same", size_bytes=12)
    record = create_record(session, content.id, "a.safetensors")

    assert session.scalar(sa.select(AssetContent).where(AssetContent.hash == "same")) == content
    assert record.content_id == content.id


def test_equal_hashes_do_not_merge_content_rows(session: Session) -> None:
    first = create_content(session, "/models/a.safetensors", hash="same")
    second = create_content(session, "/models/b.safetensors", hash="same")

    assert first.id != second.id
