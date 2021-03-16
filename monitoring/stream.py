import json
from collections import defaultdict
from dataclasses import dataclass, fields
from os import environ
from typing import Dict, List, Set, Optional

import pandas as pd
from mlrun.api.schemas.model_endpoints import ModelEndpoint
from mlrun.utils import logger
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client
from storey import (
    FieldAggregator,
    NoopDriver,
    Table,
    Source,
    Map,
    MapClass,
    AggregateByKey,
    build_flow,
    FlatMap,
    WriteToParquet,
    Batch,
    Filter,
    WriteToTSDB,
)
from storey.dtypes import SlidingWindows
from storey.steps import SampleWindow

# Constants
FUNCTION_URI = "function_uri"
MODEL = "model"
VERSION = "version"
MODEL_CLASS = "model_class"
TIMESTAMP = "timestamp"
ENDPOINT_ID = "endpoint_id"
REQUEST_ID = "request_id"
LABELS = "labels"
UNPACKED_LABELS = "unpacked_labels"
LATENCY_AVG_1S = "latency_avg_1s"
PREDICTIONS_PER_SECOND_COUNT_1S = "predictions_per_second_count_1s"
FIRST_REQUEST = "first_request"
LAST_REQUEST = "last_request"
ERROR_COUNT = "error_count"
ENTITIES = "entities"
FEATURE_NAMES = "feature_names"
PREDICTIONS_PER_SECOND = "predictions_per_second"
LATENCY = "latency"
RECORD_TYPE = "record_type"
FEATURES = "features"
PREDICTION = "prediction"
NAMED_FEATURES = "named_features"
BASE_METRICS = "base_metrics"
CUSTOM_METRICS = "custom_metrics"
ENDPOINT_FEATURES = "endpoint_features"
METRICS = "metrics"
BATCH_TIMESTAMP = "batch_timestamp"

# Configuration
@dataclass
class Config:
    sample_window: int
    kv_path_template: str
    tsdb_path_template: str
    parquet_path_template: str
    tsdb_batching_max_events: int
    tsdb_batching_timeout_secs: int
    parquet_batching_max_events: int
    parquet_batching_timeout_secs: int
    container: str
    v3io_access_key: str
    v3io_framesd: str
    time_format: str

    _environment_override: bool = True

    def __post_init__(self):
        if self._environment_override:
            for field in fields(self):
                if field.name.startswith("_"):
                    continue
                val = environ.get(field.name.upper(), field.value)
                setattr(self, field.name, val)


config = Config(
    sample_window=10,
    kv_path_template="{project}/model-endpoints/endpoints",
    tsdb_path_template="{project}/model-endpoints/events",
    parquet_path_template="/v3io/projects/{project}/model-endpoints/parquet",  # Assuming v3io is mounted
    tsdb_batching_max_events=10,
    tsdb_batching_timeout_secs=60 * 5,  # Default 5 minutes
    parquet_batching_max_events=10_000,
    parquet_batching_timeout_secs=60 * 60,  # Default 1 hour
    container="projects",
    v3io_access_key="",
    v3io_framesd="",
    time_format="%Y-%m-%d %H:%M:%S.%f",  # ISO 8061
)


# Stream processing code
class EventStreamProcessor:
    def __init__(self, project: str):

        self.project = project
        self.kv_path = config.kv_path_template.format(project=self.project)
        self.tsdb_path = config.tsdb_path_template.format(project=self.project)
        self.parquet_path = config.parquet_path_template.format(project=self.project)

        self._kv_keys = [
            FUNCTION_URI,
            MODEL,
            MODEL_CLASS,
            TIMESTAMP,
            ENDPOINT_ID,
            LABELS,
            UNPACKED_LABELS,
            LATENCY_AVG_1S,
            PREDICTIONS_PER_SECOND_COUNT_1S,
            FIRST_REQUEST,
            LAST_REQUEST,
            ERROR_COUNT,
        ]

        logger.info(
            "Writer paths",
            kv_path=self.kv_path,
            tsdb_path=self.tsdb_path,
            parquet_path=self.parquet_path,
        )

        self._flow = build_flow(
            [
                Source(),
                ProcessEndpointEvent(self.kv_path),
                FilterNotNone(),
                FlattenPredictions(),
                MapFeatureNames(self.kv_path),
                # Branch 1: Aggregate events, count averages and update TSDB and KV
                [
                    AggregateByKey(
                        aggregates=[
                            FieldAggregator(
                                PREDICTIONS_PER_SECOND,
                                ENDPOINT_ID,
                                ["count"],
                                SlidingWindows(["1s"], "1s"),
                            ),
                            FieldAggregator(
                                LATENCY, LATENCY, ["avg"], SlidingWindows(["1s"], "1s"),
                            ),
                        ],
                        table=Table("notable", NoopDriver()),
                    ),
                    SampleWindow(config.sample_window),
                    # Branch 1.1: Updated KV
                    [
                        Map(self.process_before_kv),
                        WriteToKV(table=self.kv_path),
                        InferSchema(table=self.kv_path),
                    ],
                    # Branch 1.2: Update TSDB
                    [
                        # Map the event into taggable fields, add record type to each field
                        Map(self.process_before_events_tsdb),
                        [
                            FilterKeys(BASE_METRICS),
                            UnpackValues(BASE_METRICS),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col=TIMESTAMP,
                                container=config.container,
                                access_key=config.v3io_access_key,
                                v3io_frames=config.v3io_framesd,
                                index_cols=[ENDPOINT_ID, RECORD_TYPE],
                                # Settings for _Batching
                                max_events=config.tsdb_batching_max_events,
                                timeout_secs=config.tsdb_batching_timeout_secs,
                                key=ENDPOINT_ID,
                            ),
                        ],
                        [
                            FilterKeys(ENDPOINT_FEATURES),
                            UnpackValues(ENDPOINT_FEATURES),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col=TIMESTAMP,
                                container=config.container,
                                access_key=config.v3io_access_key,
                                v3io_frames=config.v3io_framesd,
                                index_cols=[ENDPOINT_ID, RECORD_TYPE],
                                # Settings for _Batching
                                max_events=config.tsdb_batching_max_events,
                                timeout_secs=config.tsdb_batching_timeout_secs,
                                key=ENDPOINT_ID,
                            ),
                        ],
                        [
                            FilterKeys(CUSTOM_METRICS),
                            FilterNotNone(),
                            UnpackValues(CUSTOM_METRICS),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col=TIMESTAMP,
                                container=config.container,
                                access_key=config.v3io_access_key,
                                v3io_frames=config.v3io_framesd,
                                index_cols=[ENDPOINT_ID, RECORD_TYPE],
                                # Settings for _Batching
                                max_events=config.tsdb_batching_max_events,
                                timeout_secs=config.tsdb_batching_timeout_secs,
                                key=ENDPOINT_ID,
                            ),
                        ],
                    ],
                ],
                # Branch 2: Batch events, write to parquet
                [
                    Batch(
                        max_events=config.parquet_batching_max_events,
                        timeout_secs=config.parquet_batching_timeout_secs,
                        key=ENDPOINT_ID,
                    ),
                    FlatMap(lambda batch: self.process_before_parquet(batch)),
                    WriteToParquet(
                        path=self.parquet_path,
                        partition_cols=[ENDPOINT_ID, "batch_timestamp"],
                        infer_columns_from_data=True,
                        # Settings for _Batching
                        max_events=config.parquet_batching_max_events,
                        timeout_secs=config.parquet_batching_timeout_secs,
                        key=ENDPOINT_ID,
                    ),
                ],
            ]
        ).run()

    def consume(self, event: Dict):
        if "headers" in event and "values" in event:
            for values in event["values"]:
                sub_event = {k: v for k, v in zip(event["headers"], values)}
                self._flow.emit(sub_event)
        else:
            self._flow.emit(event)

    def process_before_kv(self, event: Dict):
        # Filter relevant keys
        e = {k: event[k] for k in self._kv_keys}
        # Unpack labels dictionary
        e = {**e, **e.pop(UNPACKED_LABELS, {})}
        # Write labels to kv as json string to be presentable later
        e[LABELS] = json.dumps(e[LABELS])
        return e

    @staticmethod
    def process_before_events_tsdb(event: Dict):
        base_fields = [TIMESTAMP, ENDPOINT_ID]

        base_event = {k: event[k] for k in base_fields}
        base_event[TIMESTAMP] = pd.to_datetime(
            base_event[TIMESTAMP], format=config.time_format
        )

        base_metrics = {
            RECORD_TYPE: BASE_METRICS,
            PREDICTIONS_PER_SECOND_COUNT_1S: event[PREDICTIONS_PER_SECOND_COUNT_1S],
            LATENCY_AVG_1S: event[LATENCY_AVG_1S],
            **base_event,
        }

        endpoint_features = {
            RECORD_TYPE: ENDPOINT_FEATURES,
            PREDICTION: event[PREDICTION],
            **event[NAMED_FEATURES],
            **base_event,
        }

        processed = {BASE_METRICS: base_metrics, ENDPOINT_FEATURES: endpoint_features}

        if event[METRICS]:
            processed[CUSTOM_METRICS] = {
                RECORD_TYPE: CUSTOM_METRICS,
                **event[METRICS],
                **base_event,
            }

        return processed

    @staticmethod
    def process_before_parquet(batch: List[dict]):
        def set_none_if_empty(_event: dict, keys: List[str]):
            for key in keys:
                if not _event.get(key):
                    _event[key] = None

        def drop_if_exists(_event: dict, keys: List[str]):
            for key in keys:
                _event.pop(key, None)

        if batch:
            last_event = batch[-1][TIMESTAMP]
            for event in batch:
                event[BATCH_TIMESTAMP] = last_event
                drop_if_exists(event, [UNPACKED_LABELS])
                set_none_if_empty(event, [LABELS, METRICS, ENTITIES])
        return batch


class ProcessEndpointEvent(MapClass):
    def __init__(self, kv_path: str, **kwargs):
        super().__init__(**kwargs)
        self.kv_path: str = kv_path
        self.first_request: Dict[str, str] = dict()
        self.last_request: Dict[str, str] = dict()
        self.error_count: Dict[str, int] = defaultdict(int)
        self.endpoints: Set[str] = set()

    def do(self, event: dict):
        function_uri = event.get(FUNCTION_URI)
        if not self.validate_input(function_uri, [FUNCTION_URI]):
            return None

        model = event.get(MODEL)
        if not self.validate_input(model, [MODEL]):
            return None

        version = event.get(VERSION)
        versioned_model = f"{model}_{version}" if version else model

        endpoint_id = ModelEndpoint.create_endpoint_id(
            function_uri=function_uri, versioned_model=versioned_model,
        )
        endpoint_id = str(endpoint_id)

        # In case this process fails, resume state from existing record
        self.resume_state(endpoint_id)

        # Handle errors coming from stream
        found_errors = self.handle_errors(endpoint_id, event)
        if found_errors:
            return None

        # Validate event fields
        model_class = event.get("model_class") or event.get("class")
        timestamp = event.get("when")
        request_id = event.get("request", {}).get("id")
        latency = event.get("microsec")
        features = event.get("request", {}).get("inputs")
        prediction = event.get("resp", {}).get("outputs")

        if not self.validate_input(timestamp, ["when"]):
            return None

        if endpoint_id not in self.first_request:
            self.first_request[endpoint_id] = timestamp
        self.last_request[endpoint_id] = timestamp

        if not self.validate_input(request_id, ["request", "id"]):
            return None
        if not self.validate_input(latency, ["microsec"]):
            return None
        if not self.validate_input(features, ["request", "inputs"]):
            return None
        if not self.validate_input(prediction, ["resp", "outputs"]):
            return None

        event = {
            FUNCTION_URI: function_uri,
            MODEL: model,
            MODEL_CLASS: model_class,
            TIMESTAMP: timestamp,
            ENDPOINT_ID: endpoint_id,
            REQUEST_ID: request_id,
            LATENCY: latency,
            FEATURES: features,
            PREDICTION: prediction,
            FIRST_REQUEST: self.first_request[endpoint_id],
            LAST_REQUEST: self.last_request[endpoint_id],
            ERROR_COUNT: self.error_count[endpoint_id],
            LABELS: event.get(LABELS, {}),
            METRICS: event.get(METRICS, {}),
            ENTITIES: event.get("request", {}).get(ENTITIES, {}),
            UNPACKED_LABELS: {f"_{k}": v for k, v in event.get(LABELS, {}).items()},
        }

        return event

    def resume_state(self, endpoint_id):
        # Make sure process is resumable, if process fails for any reason, be able to pick things up close to where we
        # left them
        if endpoint_id not in self.endpoints:
            logger.info("Trying to resume state", endpoint_id=endpoint_id)
            endpoint_record = get_endpoint_record(
                path=self.kv_path, endpoint_id=endpoint_id,
            )
            if endpoint_record:
                first_request = endpoint_record.get(FIRST_REQUEST)
                if first_request:
                    self.first_request[endpoint_id] = first_request
                error_count = endpoint_record.get(ERROR_COUNT)
                if error_count:
                    self.error_count[endpoint_id] = error_count
            self.endpoints.add(endpoint_id)

    def handle_errors(self, endpoint_id, event) -> bool:
        if "error" in event:
            self.error_count += 1
            return True

        return False

    def validate_input(self, field, dict_path: List[str]):
        if field is None:
            logger.error(
                f"Expected event field is missing: {field} [Event -> {''.join(dict_path)}]"
            )
            self.error_count += 1
            return False
        return True


class FlattenPredictions(FlatMap):
    def __init__(self, **kwargs):
        super().__init__(fn=FlattenPredictions.flatten, **kwargs)

    @staticmethod
    def flatten(event: Dict):
        predictions = []
        for features, prediction in zip(event[FEATURES], event[PREDICTION]):
            predictions.append(dict(event, features=features, prediction=prediction))
        return predictions


class FilterNotNone(Filter):
    def __init__(self, **kwargs):
        super().__init__(fn=lambda event: event is not None, **kwargs)


class FilterKeys(MapClass):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.keys = list(args)

    def do(self, event):
        new_event = {}
        for key in self.keys:
            if key in event:
                new_event[key] = event[key]

        return new_event if new_event else None


class UnpackValues(MapClass):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.keys_to_unpack = set(args)

    def do(self, event):
        unpacked = {}
        for key in event.keys():
            if key in self.keys_to_unpack:
                unpacked = {**unpacked, **event[key]}
            else:
                unpacked[key] = event[key]
        return unpacked


class MapFeatureNames(MapClass):
    def __init__(self, kv_path: str, **kwargs):
        super().__init__(**kwargs)
        self.kv_path = kv_path
        self.feature_names = {}

    def do(self, event: Dict):
        endpoint_id = event[ENDPOINT_ID]

        if endpoint_id not in self.feature_names:
            endpoint_record = get_endpoint_record(
                path=self.kv_path, endpoint_id=endpoint_id,
            )
            feature_names = endpoint_record.get(FEATURE_NAMES)
            feature_names = json.loads(feature_names) if feature_names else None

            if not feature_names:
                logger.warn(
                    f"Seems like endpoint {event[ENDPOINT_ID]} was not registered, feature names will be "
                    f"automatically generated"
                )
                feature_names = [f"f{i}" for i, _ in enumerate(event[FEATURES])]
                get_v3io_client().kv.update(
                    container=config.container,
                    table_path=self.kv_path,
                    key=event[ENDPOINT_ID],
                    attributes={FEATURE_NAMES: json.dumps(feature_names)},
                )

            self.feature_names[endpoint_id] = feature_names

        feature_names = self.feature_names[endpoint_id]
        features = event[FEATURES]
        event[NAMED_FEATURES] = {
            name: feature for name, feature in zip(feature_names, features)
        }
        return event


class WriteToKV(MapClass):
    def __init__(self, table: str, **kwargs):
        super().__init__(**kwargs)
        self.table = table

    def do(self, event: Dict):
        get_v3io_client().kv.update(
            container=config.container,
            table_path=self.table,
            key=event[ENDPOINT_ID],
            attributes=event,
        )
        return event


class InferSchema(MapClass):
    def __init__(self, table: str, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.keys = set()

    def do(self, event: Dict):
        key_set = set(event.keys())
        if not key_set.issubset(self.keys):
            self.keys.update(key_set)
            get_frames_client(
                token=config.v3io_access_key,
                container=config.container,
                address=config.v3io_framesd,
            ).execute(backend="kv", table=self.table, command="infer_schema")
        return event


def get_endpoint_record(path: str, endpoint_id: str) -> Optional[dict]:
    logger.info(
        f"Grabbing endpoint data", endpoint_id=endpoint_id, table_path=path,
    )
    try:
        endpoint_record = (
            get_v3io_client()
            .kv.get(container=config.container, table_path=path, key=endpoint_id,)
            .output.item
        )
        return endpoint_record
    except Exception:
        return None
