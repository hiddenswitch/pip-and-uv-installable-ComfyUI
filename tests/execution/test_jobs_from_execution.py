import pytest

from comfy_execution.graph_utils import GraphBuilder
from .common import ComfyClient, client_fixture


class TestJobs:
    client = pytest.fixture(client_fixture, scope="class", autouse=True, params=[
        {"extra_args": {}, "should_cache_results": True},
    ])

    @pytest.fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    async def _create_history_item(self, client: ComfyClient, builder: GraphBuilder):
        g = GraphBuilder(prefix="offset_test")
        input_node = g.node(
            "StubImage", content="BLACK", height=32, width=32, batch_size=1
        )
        g.node("SaveImage", images=input_node.out(0))
        return await client.run(g)

    async def test_jobs_api_job_structure(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        await self._create_history_item(client, builder)

        jobs_response = await client.get_jobs(status="completed", limit=1)
        assert len(jobs_response["jobs"]) > 0

        job = jobs_response["jobs"][0]
        assert "id" in job
        assert "status" in job
        assert "create_time" in job
        assert "outputs_count" in job
        assert "preview_output" in job

    async def test_jobs_api_preview_output_structure(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        await self._create_history_item(client, builder)

        jobs_response = await client.get_jobs(status="completed", limit=1)
        job = jobs_response["jobs"][0]

        if job["preview_output"] is not None:
            preview = job["preview_output"]
            assert "filename" in preview
            assert "nodeId" in preview
            assert "mediaType" in preview

    async def test_jobs_api_pagination(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        for _ in range(5):
            await self._create_history_item(client, builder)

        first_page = await client.get_jobs(limit=2, offset=0)
        second_page = await client.get_jobs(limit=2, offset=2)

        assert len(first_page["jobs"]) <= 2
        assert len(second_page["jobs"]) <= 2

        first_ids = {j["id"] for j in first_page["jobs"]}
        second_ids = {j["id"] for j in second_page["jobs"]}
        assert first_ids.isdisjoint(second_ids)

    async def test_jobs_api_sorting(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        for _ in range(3):
            await self._create_history_item(client, builder)

        desc_jobs = await client.get_jobs(sort_order="desc")
        asc_jobs = await client.get_jobs(sort_order="asc")

        if len(desc_jobs["jobs"]) >= 2:
            desc_times = [j["create_time"] for j in desc_jobs["jobs"] if j["create_time"]]
            asc_times = [j["create_time"] for j in asc_jobs["jobs"] if j["create_time"]]
            if len(desc_times) >= 2:
                assert desc_times == sorted(desc_times, reverse=True)
            if len(asc_times) >= 2:
                assert asc_times == sorted(asc_times)

    async def test_jobs_api_status_filter(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        await self._create_history_item(client, builder)

        completed_jobs = await client.get_jobs(status="completed")
        assert len(completed_jobs["jobs"]) > 0

        for job in completed_jobs["jobs"]:
            assert job["status"] == "completed"

        pending_jobs = await client.get_jobs(status="pending")
        for job in pending_jobs["jobs"]:
            assert job["status"] == "pending"

    async def test_get_job_by_id(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        result = await self._create_history_item(client, builder)
        prompt_id = result.get_prompt_id()

        job = await client.get_job(prompt_id)
        assert job is not None
        assert job["id"] == prompt_id
        assert "outputs" in job

    async def test_get_job_not_found(
            self, client: ComfyClient, builder: GraphBuilder
    ):
        job = await client.get_job("nonexistent-job-id")
        assert job is None
