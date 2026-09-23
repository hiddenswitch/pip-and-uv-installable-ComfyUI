from sqlalchemy.orm import Session

from comfy.app.assets.database.queries import create_content
from comfy.app.assets.database.queries import create_record
from comfy.app.assets.database.queries import delete_record
from comfy.app.assets.database.queries import get_record_by_id


def test_delete_removes_only_the_requested_delivery_record(session: Session) -> None:
    content = create_content(session, "/output/shared.png")
    first = create_record(session, content.id, "first.png")
    second = create_record(session, content.id, "second.png")

    delete_record(session, first.id)

    assert get_record_by_id(session, first.id) is None
    assert get_record_by_id(session, second.id) == second
