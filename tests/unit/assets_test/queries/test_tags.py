from sqlalchemy.orm import Session

from comfy.app.assets.database.queries import create_content
from comfy.app.assets.database.queries import create_record
from comfy.app.assets.database.queries import fetch_record_tags
from comfy.app.assets.database.queries import list_records_page
from comfy.app.assets.database.queries.records import RecordPageSpec


def test_record_tags_filter_records_without_affecting_content(session: Session) -> None:
    content = create_content(session, "/models/shared.safetensors")
    tagged = create_record(session, content.id, "tagged", tags=["model"])
    untagged = create_record(session, content.id, "untagged")

    records, _, _ = list_records_page(
        session,
        RecordPageSpec(all_tags=("model",)),
    )

    assert [record.id for record in records] == [tagged.id]
    assert fetch_record_tags(session, untagged.id) == []
