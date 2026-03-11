from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import html
import logging
import os

import aiohttp
from aiohttp import web

from ..component_model.configuration import Configuration
from ..vendor.appdirs import user_cache_dir
from .builder import FacadeWheelBuilder
from .registry import FacadeRegistry, FacadeRegistryProtocol, SnapshotFacadeRegistry

logger = logging.getLogger(__name__)


def _simple_html(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html><head><title>{html.escape(title)}</title></head><body>{body}</body></html>"


def create_facade_app(
    *,
    configuration: Configuration,
) -> web.Application:
    app = web.Application()
    app["facade_ready"] = False
    cache_prefix = configuration.pip_facade_cache_prefix
    if cache_prefix is None:
        cache_prefix = os.path.join(user_cache_dir(appname="comfyui"), "pip_facade")

    async def on_startup(application: web.Application) -> None:
        with tracer.start_as_current_span("Initialize Pip Facade Server") as span:
            span.set_attribute("facade.listen", configuration.listen)
            span.set_attribute("facade.port", configuration.port)
            span.set_attribute("facade.cache_prefix", str(cache_prefix))
            timeout = aiohttp.ClientTimeout(total=10 * 60.0, connect=60.0)
            session = aiohttp.ClientSession(timeout=timeout)
            registry: FacadeRegistryProtocol
            if configuration.pip_facade_snapshot_uri:
                registry = SnapshotFacadeRegistry(snapshot_uri=configuration.pip_facade_snapshot_uri)
                span.set_attribute("facade.snapshot_uri", configuration.pip_facade_snapshot_uri)
            else:
                registry = FacadeRegistry(
                    session,
                    base_url=configuration.pip_facade_registry_base_url,
                    only_known_nodes=configuration.pip_facade_only_known_nodes,
                )
                span.set_attribute("facade.registry_base_url", configuration.pip_facade_registry_base_url)
            builder = FacadeWheelBuilder(session, registry, cache_prefix=cache_prefix)
            application["facade_session"] = session
            application["facade_registry"] = registry
            application["facade_builder"] = builder
            with tracer.start_as_current_span("Warm Pip Facade Registry") as warmup_span:
                projects = await registry.list_projects()
                warmup_span.set_attribute("facade.project_count", len(projects))
            application["facade_ready"] = True

    async def on_cleanup(application: web.Application) -> None:
        application["facade_ready"] = False
        session: aiohttp.ClientSession | None = application.get("facade_session")
        if session is not None:
            await session.close()

    async def liveness(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, "live": True, "ready": bool(app.get("facade_ready"))})

    async def readiness(_: web.Request) -> web.Response:
        ready = bool(app.get("facade_ready"))
        status = 200 if ready else 503
        return web.json_response({"ok": ready, "live": True, "ready": ready}, status=status)

    async def index(_: web.Request) -> web.Response:
        with tracer.start_as_current_span("Serve Pip Facade Index") as span:
            registry: FacadeRegistryProtocol = app["facade_registry"]
            projects = await registry.list_projects()
            span.set_attribute("facade.project_count", len(projects))
            links = "\n".join(
                f'<a href="/simple/{project.canonical_name}/">{html.escape(project.canonical_name)}</a><br/>'
                for project in projects
            )
            return web.Response(text=_simple_html("Simple Index", links), content_type="text/html")

    async def project_page(request: web.Request) -> web.Response:
        with tracer.start_as_current_span("Serve Pip Facade Project Index") as span:
            registry: FacadeRegistryProtocol = app["facade_registry"]
            builder: FacadeWheelBuilder = app["facade_builder"]
            project = await registry.get_project(request.match_info["project"])
            if project is None:
                raise web.HTTPNotFound(text="Unknown project")
            span.set_attribute("facade.project_name", project.canonical_name)

            versions = await registry.list_versions(project)
            span.set_attribute("facade.version_count", len(versions))
            links = "\n".join(
                f'<a href="/packages/{project.canonical_name}/{item.version}/'
                f'{builder.wheel_filename(project, item.version)}">'
                f'{html.escape(builder.wheel_filename(project, item.version))}</a><br/>'
                for item in versions
            )
            return web.Response(
                text=_simple_html(f"Simple Index for {project.canonical_name}", links),
                content_type="text/html",
            )

    async def package_download(request: web.Request) -> web.StreamResponse:
        with tracer.start_as_current_span("Serve Pip Facade Wheel") as span:
            registry: FacadeRegistryProtocol = app["facade_registry"]
            builder: FacadeWheelBuilder = app["facade_builder"]

            project = await registry.get_project(request.match_info["project"])
            if project is None:
                raise web.HTTPNotFound(text="Unknown project")
            span.set_attribute("facade.project_name", project.canonical_name)

            version = await registry.get_version(project, request.match_info["version"])
            if version is None:
                raise web.HTTPNotFound(text="Unknown version")
            span.set_attribute("facade.version", version.version)

            expected_name = builder.wheel_filename(project, version.version)
            if request.match_info["filename"] != expected_name:
                raise web.HTTPNotFound(text="Filename does not match requested project/version")

            wheel = await builder.build_wheel(project, version)
            span.set_attribute("facade.wheel_path", wheel.cache_path)
            if wheel.local_path is not None:
                return web.FileResponse(path=wheel.local_path)
            return web.Response(
                body=await builder.read_cached_wheel(wheel),
                content_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{expected_name}"'},
            )

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/livez", liveness)
    app.router.add_get("/readyz", readiness)
    app.router.add_get("/healthz", readiness)
    app.router.add_get("/simple", index)
    app.router.add_get("/simple/", index)
    app.router.add_get("/simple/{project}/", project_page)
    app.router.add_get("/packages/{project}/{version}/{filename}", package_download)
    app.router.add_get("/{project}/", project_page)
    return app


async def run_facade_server(
    *,
    configuration: Configuration,
) -> None:
    with tracer.start_as_current_span("Run Pip Facade Server") as span:
        span.set_attribute("facade.listen", configuration.listen)
        span.set_attribute("facade.port", configuration.port)
        app = create_facade_app(configuration=configuration)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, configuration.listen, configuration.port)
        await site.start()
        logger.info("Serving custom-node pip facade on http://%s:%s/simple/", configuration.listen, configuration.port)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await runner.cleanup()
