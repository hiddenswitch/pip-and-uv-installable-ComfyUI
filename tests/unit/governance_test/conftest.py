import pytest


@pytest.fixture
def node_registry():
    """The shared node registry, restored after the test.

    import_all_nodes_in_workspace clears and refills the registry get_nodes() serves, and apply_disabled_nodes prunes
    it in place, so a test that does either puts the session's node set back afterwards.
    """
    from comfy.nodes import package
    from comfy.nodes_context import get_nodes

    registries = [get_nodes()]
    loaded = getattr(package._nodes_local, "nodes", None)
    if loaded is not None:
        registries.append(loaded)
    saved = [
        (registry, dict(registry.NODE_CLASS_MAPPINGS), dict(registry.NODE_DISPLAY_NAME_MAPPINGS), dict(registry.EXTENSION_WEB_DIRS))
        for registry in registries
    ]
    try:
        yield registries[0]
    finally:
        for registry, class_mappings, display_name_mappings, web_dirs in saved:
            registry.NODE_CLASS_MAPPINGS.clear()
            registry.NODE_CLASS_MAPPINGS.update(class_mappings)
            registry.NODE_DISPLAY_NAME_MAPPINGS.clear()
            registry.NODE_DISPLAY_NAME_MAPPINGS.update(display_name_mappings)
            registry.EXTENSION_WEB_DIRS.clear()
            registry.EXTENSION_WEB_DIRS.update(web_dirs)
