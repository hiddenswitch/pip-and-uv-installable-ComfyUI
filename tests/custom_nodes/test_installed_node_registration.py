"""Exercise every installed custom-node distribution through the runtime loaders."""
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib.metadata import entry_points
from importlib.resources import files

import torch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from comfy.cmd import folder_paths
from comfy.cmd.node_info import node_info
from comfy.execution_context import context_add_custom_nodes
from comfy.nodes.package import _extract_vanilla_custom_node_roots, _import_and_enumerate_nodes_in_module
from comfy.nodes.package_typing import ExportedNodes
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_2
from comfy_compatibility.vanilla import prepare_vanilla_environment


def test_installed_custom_node_registration(caplog, tmp_path):
    installed = {}
    for entry_point in entry_points(group='comfyui.custom_nodes'):
        installed.setdefault(entry_point.dist.name, []).append(entry_point)
    requirements = files('tests').joinpath('custom_nodes_requirements.txt').read_text()
    expected = {
        canonicalize_name(Requirement(line).name)
        for line in requirements.splitlines() if line.strip() and not line.startswith('#')
    }
    assert expected <= {canonicalize_name(name) for name in installed}, 'Required custom-node packages are missing'

    folder_paths.create_directories()
    prepare_vanilla_environment()
    packages = {}
    all_nodes = ExportedNodes()
    failures = []
    with caplog.at_level(logging.ERROR):
        for distribution, package_entry_points in sorted(installed.items()):
            caplog.clear()
            exported = ExportedNodes()
            for entry_point in package_entry_points:
                module = entry_point.load()
                roots = _extract_vanilla_custom_node_roots(module)
                if roots:
                    exported.update(_vanilla_load_custom_nodes_2(roots))
                else:
                    exported.update(_import_and_enumerate_nodes_in_module(module, raise_on_failure=True))
            if not exported.NODE_CLASS_MAPPINGS:
                failures.append(f'{distribution} registered no nodes')
            failures.extend(f'{distribution}: {record.getMessage()}' for record in caplog.records if record.levelno >= logging.ERROR)
            packages[distribution] = exported
            all_nodes.update(exported)

        # Schemas may refer to nodes from another package, just as /object_info
        # does after the complete startup registration phase.
        with context_add_custom_nodes(all_nodes):
            for distribution, exported in packages.items():
                caplog.clear()
                for name in exported.NODE_CLASS_MAPPINGS:
                    info = node_info(name, exported.NODE_CLASS_MAPPINGS, exported.NODE_DISPLAY_NAME_MAPPINGS)
                    assert info['name'] == name
                failures.extend(f'{distribution}: {record.getMessage()}' for record in caplog.records if record.levelno >= logging.ERROR)
                print(f'{distribution}: {len(exported.NODE_CLASS_MAPPINGS)} nodes registered with frontend schemas')  # noqa: T201
    assert not failures, '\n'.join(failures)
    assert {'ACN_ControlNetLoaderAdvanced', 'ACN_AdvancedControlNetApply_v2'} <= packages['comfyui-advanced-controlnet'].NODE_CLASS_MAPPINGS.keys()
    if 'comfyui-nunchaku' in packages:
        assert {'NunchakuPulidApply', 'NunchakuPulidLoader', 'NunchakuPuLIDLoaderV2', 'NunchakuFluxPuLIDApplyV2'} <= packages['comfyui-nunchaku'].NODE_CLASS_MAPPINGS.keys()

    if 'PixelOE' in all_nodes.NODE_CLASS_MAPPINGS:
        pixel_node = all_nodes.NODE_CLASS_MAPPINGS['PixelOE']
        inputs = {
            name: spec[0][0]
            for name, spec in pixel_node.INPUT_TYPES()['required'].items()
            if isinstance(spec[0], list)
        }
        inputs.update(
            pixel_size=4, thickness=2, img=torch.rand(1, 63, 63, 3), mode='contrast',
            color_quant=False, no_post_upscale=False, num_colors=16,
            quant_mode='kmeans', dither_mode='none', device='cpu',
        )
        result = pixel_node().execute(**inputs)
        assert len(result) == 3 and result[0].shape == (1, 64, 64, 3)
        assert all(torch.isfinite(image).all() for image in result)

    pixelized, = all_nodes.NODE_CLASS_MAPPINGS["PixelOEPixelize+"]().execute(
        image=torch.rand(1, 64, 64, 3), downscale_mode="contrast",
        target_size=16, patch_size=4, thickness=1, color_matching=False, upscale=True,
    )
    assert pixelized.shape == (1, 64, 64, 3) and torch.isfinite(pixelized).all()

    _assert_fresh_server_nodes(tmp_path, set(all_nodes.NODE_CLASS_MAPPINGS))


def _assert_fresh_server_nodes(base, expected):
    # Exercise serve's own defaults in a fresh process, rather than inheriting
    # the test execution context's runtime-installation and registration state.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log_path = base / 'server.log'
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    with log_path.open('w') as log:
        process = subprocess.Popen([
            sys.executable, '-m', 'comfy.cmd.main', 'serve', '--cpu',
            '--listen', '127.0.0.1', '--port', str(port), '--base-directory', str(base),
        ], cwd=base, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text(errors='replace')[-12000:]
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/object_info', timeout=30) as response:
                        info = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise AssertionError('Server readiness timeout: ' + log_path.read_text(errors='replace')[-12000:])
            assert expected <= info.keys(), f'Nodes missing from fresh server: {sorted(expected - info.keys())}'
            assert 'IPAdapterPlus is not installed' not in json.dumps(info)
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=30) as response:
                assert response.status == 200
                assert 'html' in response.headers.get('Content-Type', '')
            print(f'Fresh server: all {len(expected)} custom nodes and frontend available')  # noqa: T201
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
