
import pytest
from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import FolderNames, ModelPaths, SupportedExtensions
from comfy.execution_context import context_folder_names_and_paths


def test_folder_paths_interface_sanity():
    assert hasattr(folder_paths, "get_system_user_directory")
    assert hasattr(folder_paths, "get_public_user_directory")
    assert hasattr(folder_paths, "get_input_directory")
    assert hasattr(folder_paths, "extension_mimetypes_cache")
    assert callable(folder_paths.get_input_directory)
    assert callable(folder_paths.get_system_user_directory)
    assert callable(folder_paths.get_public_user_directory)


def test_supported_extensions_ior_returns_self():
    fn = FolderNames()
    fn.add(ModelPaths(["regression62_a"], supported_extensions=set()))

    proxy = fn["regression62_a"].supported_extensions
    assert isinstance(proxy, SupportedExtensions)
    result = proxy.__ior__({".onnx"})
    assert result is proxy


def test_add_model_folder_path_extensions_new_folder_name():
    with context_folder_names_and_paths(FolderNames()):
        folder_paths.add_model_folder_path(
            "regression62_detection",
            "/some/path/detection",
            extensions={".onnx"},
        )

        exts = set(folder_paths.folder_names_and_paths["regression62_detection"][1])
        assert exts == {".onnx"}


def test_add_model_folder_path_extensions_preserves_existing():
    fn = FolderNames()
    fn.add(ModelPaths(["regression62_b"], supported_extensions={".safetensors"}))

    with context_folder_names_and_paths(fn):
        folder_paths.add_model_folder_path(
            "regression62_b",
            "/some/path/b",
            extensions={".onnx"},
        )

        exts = set(folder_paths.folder_names_and_paths["regression62_b"][1])
        assert exts == {".safetensors", ".onnx"}
