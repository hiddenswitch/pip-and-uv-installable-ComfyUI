"""
Setup functions extracted from main_pre.py.

Called by Typer commands (cli.py) rather than running at import time.
"""
import ctypes
import importlib.util
import logging
import os
import shutil
import warnings

from ..cli_args_types import Configuration

logger = logging.getLogger(__name__)


def setup_environment():
    """Set env vars before torch import."""
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    if os.name == "nt":
        os.environ['MIMALLOC_PURGE_DELAY'] = '0'


def setup_warning_filters():
    if os.name == "nt":
        logging.getLogger("xformers").addFilter(
            lambda record: 'A matching Triton is not available' not in record.getMessage()
        )
    warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
    warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
    warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
    warnings.filterwarnings('ignore', category=FutureWarning, message=r'`torch\.cuda\.amp\.custom_fwd.*')
    warnings.filterwarnings("ignore", category=UserWarning, message="Please use the new API settings to control TF32 behavior.*")
    warnings.filterwarnings("ignore", message="Importing from timm.models.registry is deprecated, please import via timm.models", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated, please import via timm.layers", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Inheritance class _InstrumentedApplication from web.Application is discouraged", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="Please import `gaussian_filter` from the `scipy.ndimage` namespace; the `scipy.ndimage.filters` namespace is deprecated", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support")
    warnings.filterwarnings("ignore", category=UserWarning, message="Unsupported Windows version .* ONNX Runtime supports Windows 10 and above, only.")


def setup_logging_filters():
    log_msg_to_filter = "NOTE: Redirects are currently not supported in Windows or MacOs."
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
        lambda record: log_msg_to_filter not in record.getMessage()
    )
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    logging.getLogger("__name__").addFilter(lambda record: "setup plugin" not in record.getMessage())
    logging.getLogger("asyncio").addFilter(lambda record: 'Using selector:' not in record.getMessage())
    logging.getLogger("requests_cache").setLevel(logging.ERROR)
    logging.getLogger("fsspec").setLevel(logging.WARNING)
    logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.WARNING)
    logging.getLogger("jax").setLevel(logging.WARNING)


def setup_logging(config: Configuration):
    from ..app import logger as app_logger
    app_logger.setup_logger(config.logging_level)


def setup_cuda_devices(config: Configuration):
    if config.default_device is not None:
        default_dev = config.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ','.join(map(str, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
        os.environ['HIP_VISIBLE_DEVICES'] = str(devices)

    if config.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(config.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(config.cuda_device)
        logger.info("Set cuda device to: %s", config.cuda_device)

    if config.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if config.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = config.oneapi_device_selector
        logger.info("Set oneapi device selector to: %s", config.oneapi_device_selector)


def setup_guess_settings(config: Configuration):
    if config.guess_settings:
        from .guess_settings import apply_guess_settings
        apply_guess_settings(config)


def setup_cuda_malloc():
    from ..cmd import cuda_malloc  # pylint: disable=unused-import


def setup_tracing(config: Configuration):
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.semconv.attributes import service_attributes

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.processor.baggage import BaggageSpanProcessor, ALLOW_ALL_BAGGAGE_KEYS
    from opentelemetry.instrumentation.aiohttp_server import AioHttpServerInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

    from ..tracing_compatibility import ProgressSpanSampler
    from ..tracing_compatibility import patch_spanbuilder_set_channel

    resource = Resource.create({
        service_attributes.SERVICE_NAME: config.otel_service_name,
        service_attributes.SERVICE_VERSION: config.otel_service_version,
    })

    sampler = ProgressSpanSampler()
    provider = TracerProvider(resource=resource, sampler=sampler)

    trace.set_tracer_provider(provider)

    if config.otel_exporter_otlp_endpoint is not None:
        exporter = OTLPSpanExporter()
    else:
        exporter = SpanExporter()

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    metrics_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if metrics_endpoint:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=metrics_endpoint),
            export_interval_millis=10000
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

    patch_spanbuilder_set_channel()

    AioPikaInstrumentor().instrument()
    AioHttpServerInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    URLLib3Instrumentor().instrument()

    provider.add_span_processor(BaggageSpanProcessor(ALLOW_ALL_BAGGAGE_KEYS))


def setup_fsspec():
    import fsspec
    from . import package_filesystem
    fsspec.register_implementation(
        package_filesystem.PkgResourcesFileSystem.protocol,
        package_filesystem.PkgResourcesFileSystem,
    )


def fix_pytorch_240():
    """Fixes pytorch 2.4.0 libomp issue on Windows."""
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.submodule_search_locations is None:
        return
    for folder in torch_spec.submodule_search_locations:
        lib_folder = os.path.join(folder, "lib")
        test_file = os.path.join(lib_folder, "fbgemm.dll")
        dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
        if os.path.exists(dest):
            break

        try:
            with open(test_file, 'rb') as f:
                contents = f.read()
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                _ = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                logger.warning("Detected pytorch version with libomp issue, trying to patch")
                try:
                    shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
                except Exception as exc_info:
                    logger.error("While trying to patch a fix for torch 2.4.0, an error occurred", exc_info=exc_info)
        except Exception:
            pass


def setup_pre_torch(config: Configuration):
    """Must be called before torch import."""
    setup_environment()
    setup_guess_settings(config)
    setup_cuda_devices(config)
    setup_cuda_malloc()


def setup_post_torch(config: Configuration):
    setup_warning_filters()
    setup_logging_filters()
    setup_logging(config)
    setup_fsspec()
    fix_pytorch_240()
    from .. import torchvision_compat  # pylint: disable=unused-import
    setup_tracing(config)
