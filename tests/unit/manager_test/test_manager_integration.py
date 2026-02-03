"""
Tests for comfyui_manager integration.

Verifies that manager REST API endpoints are properly injected and reachable
when --enable-manager is set, and that they are NOT available when disabled.
"""
import pytest
import requests
from multiprocessing import Process


# Manager v2 API endpoints that should be available when manager is enabled
MANAGER_ENDPOINTS = [
    # Basic status endpoints (safe to call, don't modify state)
    ("/v2/manager/queue/status", "GET"),
    ("/v2/manager/version", "GET"),
    ("/v2/manager/db_mode", "GET"),
    ("/v2/manager/is_legacy_manager_ui", "GET"),
    ("/v2/customnode/installed", "GET"),
    ("/v2/manager/channel_url_list", "GET"),
]

# Endpoints that require specific setup or have side effects - just check they exist
MANAGER_ENDPOINTS_EXISTENCE_ONLY = [
    ("/v2/customnode/getmappings", "GET"),
    ("/v2/snapshot/getlist", "GET"),
    ("/v2/snapshot/get_current", "GET"),
]


class TestManagerEnabled:
    """Tests with manager enabled."""

    def test_manager_endpoints_reachable(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify manager API endpoints are reachable when enabled."""
        base_url, _proc = manager_enabled_server

        for endpoint, method in MANAGER_ENDPOINTS:
            url = base_url + endpoint
            if method == "GET":
                response = http.get(url, timeout=30)
            else:
                response = http.post(url, timeout=30)

            # Manager endpoints should return 200 or valid JSON error (not 404)
            assert response.status_code != 404, (
                f"Manager endpoint {endpoint} returned 404 - not properly injected"
            )

    def test_manager_version_endpoint(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify /v2/manager/version returns version info."""
        base_url, _proc = manager_enabled_server
        response = http.get(f"{base_url}/v2/manager/version", timeout=30)

        assert response.status_code == 200
        # Response may be empty, a string, or JSON with version info
        if response.text:
            try:
                data = response.json()
                assert "version" in data or isinstance(data, str)
            except requests.exceptions.JSONDecodeError:
                # Plain text version string is also acceptable
                assert len(response.text) > 0

    def test_manager_queue_status(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify /v2/manager/queue/status returns queue status."""
        base_url, _proc = manager_enabled_server
        response = http.get(f"{base_url}/v2/manager/queue/status", timeout=30)

        assert response.status_code == 200
        data = response.json()
        # Should return some queue status structure
        assert isinstance(data, dict)

    def test_manager_db_mode(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify /v2/manager/db_mode returns database mode."""
        base_url, _proc = manager_enabled_server
        response = http.get(f"{base_url}/v2/manager/db_mode", timeout=30)

        # 200 status with any response (possibly empty) is acceptable
        assert response.status_code == 200

    def test_customnode_installed(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify /v2/customnode/installed returns installed nodes list."""
        base_url, _proc = manager_enabled_server
        response = http.get(f"{base_url}/v2/customnode/installed", timeout=30)

        assert response.status_code == 200
        data = response.json()
        # Should return a list of installed nodes
        assert isinstance(data, (list, dict))

    def test_manager_is_legacy_ui(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify /v2/manager/is_legacy_manager_ui returns boolean or dict with bool."""
        base_url, _proc = manager_enabled_server
        response = http.get(f"{base_url}/v2/manager/is_legacy_manager_ui", timeout=30)

        assert response.status_code == 200
        data = response.json()
        # API returns either bool directly or dict with 'is_legacy_manager_ui' key
        if isinstance(data, dict):
            assert "is_legacy_manager_ui" in data
            assert isinstance(data["is_legacy_manager_ui"], bool)
        else:
            assert isinstance(data, bool)


class TestManagerDisabled:
    """Tests with manager disabled."""

    def test_manager_endpoints_not_found(
        self,
        http: requests.Session,
        manager_disabled_server: tuple[str, Process],
    ):
        """Verify manager API endpoints return 404 when disabled."""
        base_url, _proc = manager_disabled_server

        for endpoint, method in MANAGER_ENDPOINTS:
            url = base_url + endpoint
            if method == "GET":
                response = http.get(url, timeout=30)
            else:
                response = http.post(url, timeout=30)

            # Manager endpoints should NOT be available (404)
            assert response.status_code == 404, (
                f"Manager endpoint {endpoint} should return 404 when manager disabled, "
                f"got {response.status_code}"
            )

    def test_comfyui_base_endpoints_still_work(
        self,
        http: requests.Session,
        manager_disabled_server: tuple[str, Process],
    ):
        """Verify base ComfyUI endpoints still work when manager is disabled."""
        base_url, _proc = manager_disabled_server

        # These are core ComfyUI endpoints that should always work
        core_endpoints = [
            "/system_stats",
            "/object_info",
            "/queue",
        ]

        for endpoint in core_endpoints:
            response = http.get(f"{base_url}{endpoint}", timeout=30)
            assert response.status_code == 200, (
                f"Core endpoint {endpoint} should work, got {response.status_code}"
            )


class TestManagerWithDisabledCustomNodes:
    """Tests with manager enabled but custom nodes disabled."""

    def test_manager_works_with_disabled_custom_nodes(
        self,
        http: requests.Session,
        manager_with_disabled_custom_nodes_server: tuple[str, Process],
    ):
        """Verify manager endpoints work when --disable-all-custom-nodes is set."""
        base_url, _proc = manager_with_disabled_custom_nodes_server

        # Manager endpoints should still work
        for endpoint, method in MANAGER_ENDPOINTS:
            url = base_url + endpoint
            if method == "GET":
                response = http.get(url, timeout=30)
            else:
                response = http.post(url, timeout=30)

            assert response.status_code != 404, (
                f"Manager endpoint {endpoint} returned 404 with --disable-all-custom-nodes"
            )

    def test_core_endpoints_work_with_disabled_custom_nodes(
        self,
        http: requests.Session,
        manager_with_disabled_custom_nodes_server: tuple[str, Process],
    ):
        """Verify core ComfyUI endpoints still work."""
        base_url, _proc = manager_with_disabled_custom_nodes_server

        core_endpoints = ["/system_stats", "/object_info", "/queue"]
        for endpoint in core_endpoints:
            response = http.get(f"{base_url}{endpoint}", timeout=30)
            assert response.status_code == 200, (
                f"Core endpoint {endpoint} should work, got {response.status_code}"
            )


class TestManagerMiddleware:
    """Tests for manager middleware functionality."""

    def test_middleware_allows_normal_requests(
        self,
        http: requests.Session,
        manager_enabled_server: tuple[str, Process],
    ):
        """Verify middleware doesn't block normal ComfyUI requests."""
        base_url, _proc = manager_enabled_server

        # Normal ComfyUI endpoints should work through the middleware
        response = http.get(f"{base_url}/system_stats", timeout=30)
        assert response.status_code == 200

        response = http.get(f"{base_url}/object_info", timeout=30)
        assert response.status_code == 200

        response = http.get(f"{base_url}/queue", timeout=30)
        assert response.status_code == 200
