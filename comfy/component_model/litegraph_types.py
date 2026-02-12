"""Types for LiteGraph UI workflow format."""
from __future__ import annotations

from typing import NamedTuple, Optional


class LiteLink(NamedTuple):
    """A single link in a LiteGraph workflow: [link_id, src_node, src_slot, dst_node, dst_slot, type]."""
    link_id: int
    src_node: int
    src_slot: int
    dst_node: int
    dst_slot: int
    type: Optional[str] = None

    @classmethod
    def from_list(cls, raw: list) -> LiteLink:
        return cls(
            link_id=raw[0],
            src_node=raw[1],
            src_slot=raw[2],
            dst_node=raw[3],
            dst_slot=raw[4],
            type=raw[5] if len(raw) > 5 else None,
        )

    @classmethod
    def from_dict(cls, raw: dict) -> LiteLink:
        return cls(
            link_id=raw["id"],
            src_node=raw["origin_id"],
            src_slot=raw["origin_slot"],
            dst_node=raw["target_id"],
            dst_slot=raw["target_slot"],
            type=raw.get("type"),
        )
