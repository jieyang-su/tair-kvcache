import json
import os

from hisim.spec import ModelInfo, AcceleratorInfo, DataType
from hisim.simulation.types import PlatformConfig, SchedulerConfig
from hisim.simulation.manager import Envs
from hisim.simulation.utils import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from hisim.time_predictor import (
    InferTimePredictor,
    AIConfiguratorTimePredictor,
)
from hisim.utils import get_logger


logger = get_logger()


class ConfigManager:
    _model_info: ModelInfo = None
    _platform_config: PlatformConfig = None
    _scheduler_config: SchedulerConfig = None

    @classmethod
    def set_model_info(cls, model: ModelInfo):
        cls._model_info = model

    @classmethod
    def get_model_info(
        cls, hf_config: dict | None, source_model_path: str | None = None
    ) -> ModelInfo:
        if hf_config is not None:
            model = ModelInfo.from_config(hf_config)
            if model is None:
                logger.error(
                    f"Failed to initialize model information with configuration: {hf_config}"
                )
        else:
            with open(Envs.config_path()) as f:
                config: dict = json.load(f)
            model = ModelInfo.find_by_model_name(config.get("model", {}).get("name"))

        if model is not None and source_model_path:
            original_model_path = source_model_path
            hf_model_path = model.model_path or None
            if hf_model_path != original_model_path:
                logger.info(
                    "Override model_path for simulation: hf_config_path=%s source_model_path=%s",
                    hf_model_path,
                    original_model_path,
                )

            model.model_path = original_model_path

            if os.path.isabs(original_model_path) and not os.path.exists(original_model_path):
                logger.warning(
                    "Source model_path does not exist on the current worker: %s",
                    original_model_path,
                )

        return model

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        platform_config = config.get("platform", {})
        device_name = platform_config.get("accelerator", {}).get("name")
        hw = AcceleratorInfo.find_by_hw_name(device_name)
        if hw is None:
            logger.error(
                f"Failed to initialize device info with {device_name}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            raise ValueError(f"Failed to initialize device info with {device_name}")
        else:
            logger
        return hw

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform_config is None:
            hw = cls.get_accelerator_info()
            with open(Envs.config_path()) as f:
                config: dict = json.load(f)
            platform_config = config.get("platform", {})
            cls._platform_config = PlatformConfig(
                device=hw,
                disk_read_bandwidth_gb=platform_config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=platform_config.get("disk_write_bandwidth_gb"),
                memory_read_bandwidth_gb=platform_config.get(
                    "memory_read_bandwidth_gb"
                ),
                memory_write_bandwidth_gb=platform_config.get(
                    "memory_write_bandwidth_gb"
                ),
                num_device_per_node=platform_config.get("num_device_per_node"),
            )

            logger.info(
                f"Platform configuration initialized successfully. {cls._platform_config}"
            )

        return cls._platform_config

    @classmethod
    def set_scheduler_config(cls, config: SchedulerConfig):
        cls._scheduler_config = config

    @classmethod
    def get_kv_cache_bytes(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_cell_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_kv_cache_bytes_per_layer(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_per_layer_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_scheduler_config(
        cls, server_args: dict, backend: str, hf_config: dict | None = None
    ):
        model = ConfigManager.get_model_info(
            hf_config, server_args.get("model_path")
        )

        internal_config = cls._parse_server_args(server_args, backend)

        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        scheduler_config = config.get("scheduler", {})

        tp_size = scheduler_config.get("tp_size")
        if tp_size is None:
            tp_size = internal_config.tp_size
        ep_size = scheduler_config.get("ep_size")
        if ep_size is None:
            ep_size = internal_config.ep_size
        dp_size = scheduler_config.get("dp_size")
        if dp_size is None:
            dp_size = internal_config.dp_size
        moe_tp_size = scheduler_config.get("moe_tp_size")
        enable_wideep = bool(scheduler_config.get("enable_wideep", False))
        enable_eplb = bool(scheduler_config.get("enable_eplb", False))
        moe_backend = scheduler_config.get("moe_backend")
        def parse_dtype(name: str, default=None):
            dtype = scheduler_config.get(name)
            if dtype is not None:
                return DataType(dtype.upper())
            return default

        dtype = parse_dtype("data_type")
        if dtype is None:
            dtype = DataType.from_torch_dtype(model.torch_dtype)

        gemm_dtype = parse_dtype("gemm_data_type", dtype)
        moe_dtype = parse_dtype("moe_data_type", dtype)
        kv_cache_dtype = parse_dtype("kv_cache_data_type", dtype)
        fmha_dtype = parse_dtype("fmha_data_type", kv_cache_dtype)
        comm_dtype = parse_dtype("comm_data_type", dtype)

        sched_config = SchedulerConfig(
            model=model,
            max_prefill_tokens=internal_config.max_prefill_tokens,
            chunked_prefill_size=internal_config.chunked_prefill_size,
            mem_fraction_static=internal_config.mem_fraction_static,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            num_hidden_layers=scheduler_config.get("num_hidden_layers", 0),
            moe_tp_size=moe_tp_size,
            workload_distribution=scheduler_config.get("workload_distribution", "recorded"),
            attention_backend=scheduler_config.get("attention_backend", "flashinfer"),
            enable_eplb=enable_eplb,
            enable_wideep=enable_wideep,
            moe_backend=moe_backend,
            # TODO: initialize with the runtime data type.
            data_type=dtype,
            gemm_data_type=gemm_dtype,
            moe_data_type=moe_dtype,
            kv_cache_data_type=kv_cache_dtype,
            fmha_data_type=fmha_dtype,
            comm_data_type=comm_dtype,
            page_size=internal_config.page_size,
            backend_name=backend,
            backend_version=scheduler_config.get("backend_version"),
        )
        return sched_config

    @classmethod
    def _parse_server_args(cls, server_args: dict, backend: str) -> SchedulerConfig:
        if backend == "sglang":
            return SchedulerConfig(
                model=None,
                tp_size=server_args.get("tp_size", 1),
                ep_size=server_args.get("ep_size", 1),
                dp_size=server_args.get("dp_size", 1),
                num_hidden_layers=server_args.get("num_hidden_layers", 0),
                moe_tp_size=server_args.get("moe_tp_size"),
                enable_wideep=False,
                moe_backend=None,
                max_prefill_tokens=server_args.get("max_prefill_tokens"),
                chunked_prefill_size=server_args.get("chunked_prefill_size"),
                mem_fraction_static=server_args.get("mem_fraction_static"),
                page_size=server_args.get("page_size"),
                backend_name="sglang",
            )
        else:
            raise RuntimeError(f"Unsupported backend[{backend}] server args parser.")

    @classmethod
    def get_inference_time_predictor(
        cls, model: ModelInfo, hw: AcceleratorInfo, sched_config: SchedulerConfig
    ) -> InferTimePredictor:
        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        predictor_config = config.get("predictor", {})
        if predictor_config.get("name") == "aiconfigurator":
            device_name = predictor_config.get("device_name")
            hw.name = device_name
            database_mode = predictor_config.get("database_mode", "SILICON")
            prefill_scale_factor = predictor_config.get("prefill_scale_factor", 1)
            decode_scale_factor = predictor_config.get("decode_scale_factor", 1)
            xgb_model_path = predictor_config.get("xgb_model_path", None)
            return AIConfiguratorTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                database_mode=database_mode,
                prefill_scale_factor=prefill_scale_factor,
                decode_scale_factor=decode_scale_factor,
                xgb_model_path=xgb_model_path,
            )
        else:
            raise ValueError(f"Unknown predictor name: {predictor_config.get('name')}")
