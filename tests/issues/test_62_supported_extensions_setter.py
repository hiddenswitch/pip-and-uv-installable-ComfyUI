from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import FolderNames, ModelPaths
from comfy.execution_context import context_folder_names_and_paths


def test_issue_62_extensions_kwarg_populates_new_folder():
    with context_folder_names_and_paths(FolderNames()):
        folder_paths.add_model_folder_path(
            "detection",
            "/some/path/detection",
            extensions={".onnx"},
        )

        paths, exts = folder_paths.folder_names_and_paths["detection"]
        assert "/some/path/detection" in list(paths)
        assert set(exts) == {".onnx"}


def test_issue_62_extensions_kwarg_does_not_wipe_existing():
    fn = FolderNames()
    fn.add(ModelPaths(["detection"], supported_extensions={".safetensors"}))

    with context_folder_names_and_paths(fn):
        folder_paths.add_model_folder_path(
            "detection",
            "/some/path/detection",
            extensions={".onnx"},
        )

        _, exts = folder_paths.folder_names_and_paths["detection"]
        assert set(exts) == {".safetensors", ".onnx"}


def test_issue_62_get_filename_list_returns_matching_files(tmp_path):
    detection_dir = tmp_path / "detection"
    detection_dir.mkdir()
    (detection_dir / "yolov10m.onnx").touch()
    (detection_dir / "readme.txt").touch()

    with context_folder_names_and_paths(FolderNames(base_paths=[tmp_path])):
        folder_paths.add_model_folder_path(
            "detection",
            str(detection_dir),
            extensions={".onnx"},
        )

        listed = folder_paths.get_filename_list("detection")
        assert "yolov10m.onnx" in listed
        assert "readme.txt" not in listed
