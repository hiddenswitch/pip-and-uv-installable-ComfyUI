import json
from importlib import resources

import pytest

from comfy.app.subgraph_manager import BLUEPRINTS_PACKAGE, Source, SubgraphManager


def test_blueprints_are_packaged_importlib_resources():
    blueprint_root = resources.files(BLUEPRINTS_PACKAGE)
    names = {entry.name for entry in blueprint_root.iterdir()}

    assert "__init__.py" in names
    assert "Text to Image.json" in names
    assert "put_blueprints_here" in names

    data = blueprint_root.joinpath("Text to Image.json").read_text(encoding="utf-8")
    assert json.loads(data)


@pytest.mark.asyncio
async def test_subgraph_manager_loads_blueprints_from_package_resources():
    manager = SubgraphManager()

    entries = await manager.get_blueprint_subgraphs(force_reload=True)

    assert entries
    text_to_image = next(entry for entry in entries.values() if entry["name"] == "Text to Image")
    assert text_to_image["source"] == Source.templates
    assert text_to_image["path"] == f"{BLUEPRINTS_PACKAGE}/Text to Image.json"

    await manager.load_entry_data(text_to_image)
    assert json.loads(text_to_image["data"])
