"""jobs sub-app: list and cancel server jobs (client only, no torch)."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .server_connection import fetch_json, post_json

jobs_app = typer.Typer(name="jobs", no_args_is_help=False, add_completion=False)


@jobs_app.callback(invoke_without_command=True)
def jobs_default(
    ctx: typer.Context,
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status: pending, in_progress, completed, failed."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max jobs to return."),
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL (default: http://localhost:8188)."),
):
    """List jobs on the server."""
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(_list_jobs(status=status, limit=limit, format=format, server=server))


async def _list_jobs(status: Optional[str], limit: Optional[int], format: str, server: Optional[str]):
    params = {}
    if status is not None:
        params["status"] = status
    if limit is not None:
        params["limit"] = str(limit)
    data = await fetch_json(server, "/api/jobs", params=params)
    jobs = data if isinstance(data, list) else data.get("jobs", [])

    if format == "json":
        Console().print_json(json.dumps(jobs))
        return

    console = Console()
    if not jobs:
        console.print("No jobs found.")
        return
    table = Table(show_edge=False, pad_edge=False, box=None)
    table.add_column("ID", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Duration", no_wrap=True)
    for job in jobs:
        table.add_row(
            job.get("prompt_id", job.get("id", "")),
            job.get("status", ""),
            job.get("created_at", ""),
            str(job.get("execution_duration", "")),
        )
    console.print(table)


@jobs_app.command(name="cancel")
def jobs_cancel(
    job_id: Optional[str] = typer.Argument(None, help="Job ID to cancel. If omitted, interrupts current execution."),
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL."),
):
    """Cancel a job or interrupt current execution."""
    asyncio.run(_cancel_job(job_id=job_id, server=server))


async def _cancel_job(job_id: Optional[str], server: Optional[str]):
    body = {}
    if job_id is not None:
        body["prompt_id"] = job_id
    await post_json(server, "/interrupt", body=body)
    console = Console()
    if job_id:
        console.print(f"Requested cancellation of job {job_id}")
    else:
        console.print("Requested interrupt of current execution")
