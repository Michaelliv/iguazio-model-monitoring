from os import environ
from typing import Any


class Config:
    def __init__(self, base_config: dict, environment_override: bool = True):
        self.base_config = base_config
        self.environment_override = environment_override

        for k, v in self.base_config.items():
            if environment_override:
                val = environ.get(k, v)
                setattr(self, k, environ.get(k, v))
                self.base_config[k] = val
            else:
                setattr(self, k, v)

    def get(
        self,
        key: str,
        integer_number: bool = False,
        float_number: bool = False,
        default: Any = None,
    ):
        if integer_number and float_number:
            raise RuntimeError(
                "Both 'integer_number' and 'float_number' are set to 'True'"
            )

        value = self.base_config.get(key)
        if value is None:
            if default is not None:
                return default
            else:
                raise KeyError(f"'{key}' not found in config")

        if integer_number:
            return int(value)

        if float_number:
            return float(value)

        return value

    def get_int(self, key: str, default: int = None):
        return self.get(key, integer_number=True, default=default)

    def get_float(self, key: str, default: float = None):
        return self.get(key, float_number=True, default=default)


config = Config(
    {
        "SAMPLE_WINDOW": 10,
        "KV_PATH_TEMPLATE": "{project}/model-endpoints/endpoints",
        "TSDB_PATH_TEMPLATE": "{project}/model-endpoints/events",
        "PARQUET_PATH_TEMPLATE": "/v3io/projects/{project}/model-endpoints/parquet",  # Assuming v3io is mounted
        "TSDB_BATCHING_MAX_EVENTS": 10,
        "TSDB_BATCHING_TIMEOUT_SECS": 60 * 5,  # Default 5 minutes
        "PARQUET_BATCHING_MAX_EVENTS": 10_000,
        "PARQUET_BATCHING_TIMEOUT_SECS": 60 * 60,  # Default 1 hour
        "CONTAINER": "projects",
        "V3IO_ACCESS_KEY": "",
        "V3IO_FRAMESD": "",
        "TIME_FORMAT": "%Y-%m-%d %H:%M:%S.%f",  # ISO 8061
    }
)
