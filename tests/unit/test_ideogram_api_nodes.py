import pytest

from comfy_api_nodes.apis.ideogram import IdeogramV4Request
from comfy_api_nodes.nodes_ideogram import IdeogramExtension, IdeogramV4


def test_ideogram_v4_request_accepts_text_prompt_payload():
    request = IdeogramV4Request(
        text_prompt="a poster with bold lettering",
        resolution="2048x2048",
        rendering_speed="QUALITY",
    )

    assert request.text_prompt == "a poster with bold lettering"
    assert request.json_prompt is None
    assert request.resolution == "2048x2048"
    assert request.rendering_speed == "QUALITY"


def test_ideogram_v4_schema_exposes_partner_api_node():
    schema = IdeogramV4.define_schema()

    assert schema.node_id == "IdeogramV4"
    assert schema.display_name == "Ideogram V4"
    assert schema.category == "partner/image/Ideogram"
    assert schema.is_api_node is True
    assert len(schema.outputs) == 1
    assert schema.outputs[0].get_io_type() == "IMAGE"


@pytest.mark.asyncio
async def test_ideogram_extension_registers_v4_node():
    nodes = await IdeogramExtension().get_node_list()

    assert IdeogramV4 in nodes
