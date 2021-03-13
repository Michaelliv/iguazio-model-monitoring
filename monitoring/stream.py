import asyncio
import json
import os
from collections import defaultdict
from functools import partial
from typing import Dict, List, Set, Optional, Union, Any

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
from storey.utils import url_to_file_system

from monitoring.config import config


class EventStreamProcessor:
    def __init__(self, project: str):

        self.project = project
        self.kv_path = config.get("KV_PATH_TEMPLATE").format(project=self.project)
        self.tsdb_path = config.get("TSDB_PATH_TEMPLATE").format(project=self.project)
        self.parquet_path = config.get("PARQUET_PATH_TEMPLATE").format(
            project=self.project
        )

        self._kv_keys = [
            "timestamp",
            "project",
            "model",
            "function",
            "tag",
            "model_class",
            "endpoint_id",
            "labels",
            "unpacked_labels",
            "latency_avg_1s",
            "predictions_per_second_count_1s",
            "first_request",
            "last_request",
            "error_count",
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
                                "predictions_per_second",
                                "endpoint_id",
                                ["count"],
                                SlidingWindows(["1s"], "1s"),
                            ),
                            FieldAggregator(
                                "latency",
                                "latency",
                                ["avg"],
                                SlidingWindows(["1s"], "1s"),
                            ),
                        ],
                        table=Table("notable", NoopDriver()),
                    ),
                    SampleWindow(config.get_int("SAMPLE_WINDOW")),
                    # Branch 1.1: Updated KV
                    [
                        Map(self.process_before_kv),
                        WriteToKV(table=self.kv_path),
                        InferSchema(table=self.kv_path),
                    ],
                    # Branch 1.2: Update TSDB
                    # base_metrics
                    # endpoint_features
                    [
                        # Map the event into taggable fields, add record type to each field
                        Map(self.process_before_events_tsdb),
                        [
                            FilterKeys("base_metrics"),
                            UnpackValues("base_metrics"),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col="timestamp",
                                container=config.get("CONTAINER"),
                                access_key=config.get("V3IO_ACCESS_KEY"),
                                v3io_frames=config.get("V3IO_FRAMESD"),
                                index_cols=["endpoint_id", "record_type"],
                                # Settings for _Batching
                                max_events=config.get_int("TSDB_BATCHING_MAX_EVENTS"),
                                timeout_secs=config.get_int(
                                    "TSDB_BATCHING_TIMEOUT_SECS"
                                ),
                                key="endpoint_id",
                            ),
                        ],
                        [
                            FilterKeys("endpoint_features"),
                            UnpackValues("endpoint_features"),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col="timestamp",
                                container=config.get("CONTAINER"),
                                access_key=config.get("V3IO_ACCESS_KEY"),
                                v3io_frames=config.get("V3IO_FRAMESD"),
                                index_cols=["endpoint_id", "record_type"],
                                # Settings for _Batching
                                max_events=config.get_int("TSDB_BATCHING_MAX_EVENTS"),
                                timeout_secs=config.get_int(
                                    "TSDB_BATCHING_TIMEOUT_SECS"
                                ),
                                key="endpoint_id",
                            ),
                        ],
                        [
                            FilterKeys("custom_metrics"),
                            FilterNotNone(),
                            UnpackValues("custom_metrics"),
                            WriteToTSDB(
                                path=self.tsdb_path,
                                rate="10/m",
                                time_col="timestamp",
                                container=config.get("CONTAINER"),
                                access_key=config.get("V3IO_ACCESS_KEY"),
                                v3io_frames=config.get("V3IO_FRAMESD"),
                                index_cols=["endpoint_id", "record_type"],
                                # Settings for _Batching
                                max_events=config.get_int("TSDB_BATCHING_MAX_EVENTS"),
                                timeout_secs=config.get_int(
                                    "TSDB_BATCHING_TIMEOUT_SECS"
                                ),
                                key="endpoint_id",
                            ),
                        ],
                    ],
                ],
                # Branch 2: Batch events, write to parquet
                [
                    Batch(
                        max_events=config.get_int("PARQUET_BATCHING_MAX_EVENTS"),
                        timeout_secs=config.get_int("PARQUET_BATCHING_TIMEOUT_SECS"),
                        key="endpoint_id",
                    ),
                    FlatMap(lambda batch: _process_before_parquet(batch)),
                    WriteToParquet(
                        path=self.parquet_path,
                        partition_cols=["endpoint_id", "batch_timestamp"],
                        infer_columns_from_data=True,
                        # Settings for _Batching
                        max_events=config.get_int("PARQUET_BATCHING_MAX_EVENTS"),
                        timeout_secs=config.get_int("PARQUET_BATCHING_TIMEOUT_SECS"),
                        key="endpoint_id",
                    ),
                ],
            ]
        ).run()

    def consume(self, event: Dict):
        self._flow.emit(event)

    @staticmethod
    def unpack_predictions(event: Dict) -> List[Dict]:
        predictions = []
        for features, prediction in zip(event["features"], event["prediction"]):
            predictions.append(dict(event, features=features, prediction=prediction))
        return predictions

    def process_before_kv(self, event: Dict):
        # Filter relevant keys
        e = {k: event[k] for k in self._kv_keys}
        # Unpack labels dictionary
        e = {**e, **e.pop("unpacked_labels", {})}
        # Write labels to kv as json string to be presentable later
        e["labels"] = json.dumps(e["labels"])
        return e

    def process_before_events_tsdb(self, event: Dict):
        base_fields = [
            "timestamp",
            "endpoint_id",
        ]

        base_event = {k: event[k] for k in base_fields}
        base_event["timestamp"] = pd.to_datetime(
            base_event["timestamp"], format=config.get("TIME_FORMAT")
        )

        base_metrics = {
            "record_type": "base_metrics",
            "predictions_per_second_count_1s": event["predictions_per_second_count_1s"],
            "latency_avg_1s": event["latency_avg_1s"],
            **base_event,
        }

        endpoint_features = {
            "record_type": "endpoint_features",
            "prediction": event["prediction"],
            **event["named_features"],
            **base_event,
        }

        processed = {
            "base_metrics": base_metrics,
            "endpoint_features": endpoint_features,
        }

        if event["metrics"]:
            processed["custom_metrics"] = {
                "record_type": "custom_metrics",
                **event["metrics"],
                **base_event,
            }

        return processed


class ProcessEndpointEvent(MapClass):
    def __init__(self, kv_path: str, **kwargs):
        super().__init__(**kwargs)
        self.kv_path: str = kv_path
        self.first_request: Dict[str, str] = dict()
        self.last_request: Dict[str, str] = dict()
        self.error_count: Dict[str, int] = defaultdict(int)
        self.endpoints: Set[str] = set()

    def do(self, event: dict):
        endpoint_spec = self.endpoint_spec_from_event(event)
        endpoint_id = ModelEndpoint.create_endpoint_id(
            project=endpoint_spec["project"],
            function=endpoint_spec["function"],
            model=endpoint_spec["model"],
            tag=endpoint_spec["tag"],
        )

        # In case this process fails, resume state from existing record
        self.resume_state(endpoint_id)

        # Handle errors coming from stream
        found_errors = self.handle_errors(endpoint_id, event)
        if found_errors:
            return None

        # Validate event fields
        timestamp = event.get("when")
        request_id = event.get("request", {}).get("id")
        latency = event.get("microsec")
        features = event.get("request", {}).get("inputs")
        prediction = event.get("resp", {}).get("outputs")

        if not self.validate_input(timestamp, "when"):
            return None

        if endpoint_id not in self.first_request:
            self.first_request[endpoint_id] = timestamp
        self.last_request[endpoint_id] = timestamp

        if not self.validate_input(request_id, "request", "id"):
            return None
        if not self.validate_input(latency, "microsec"):
            return None
        if not self.validate_input(features, "request", "inputs"):
            return None
        if not self.validate_input(prediction, "resp", "outputs"):
            return None

        event = {
            "timestamp": timestamp,
            "endpoint_id": endpoint_id,
            "request_id": request_id,
            "latency": latency,
            "features": features,
            "prediction": prediction,
            "first_request": self.first_request[endpoint_id],
            "last_request": self.last_request[endpoint_id],
            "error_count": self.error_count[endpoint_id],
            "labels": event.get("labels", {}),
            "metrics": event.get("metrics", {}),
            "unpacked_labels": {f"_{k}": v for k, v in event.get("labels", {}).items()},
            **endpoint_spec,
        }

        return event

    @staticmethod
    def endpoint_spec_from_event(event: Dict) -> Dict[str, Any]:
        project, function_with_tag = event["function_uri"].split("/")

        try:
            function, tag = function_with_tag.split(":")
        except ValueError:
            function, tag = function_with_tag, ""

        model = event["model"]
        model_class = event.get("model_class") or event.get("class")

        return {
            "project": project,
            "model": model,
            "function": function,
            "tag": tag,
            "model_class": model_class,
        }

    def resume_state(self, endpoint_id):
        # Make sure process is resumable, if process fails for any reason, be able to pick things up close to where we
        # left them
        if endpoint_id not in self.endpoints:
            endpoint_record = get_endpoint_record(
                path=self.kv_path, endpoint_id=endpoint_id,
            )
            if endpoint_record:
                first_request = endpoint_record.get("first_request")
                if first_request:
                    self.first_request[endpoint_id] = first_request
                error_count = endpoint_record.get("error_count")
                if error_count:
                    self.error_count[endpoint_id] = error_count
            self.endpoints.add(endpoint_id)

    def handle_errors(self, endpoint_id, event) -> bool:
        if "error" in event:
            self.error_count += 1
            return True

        return False

    def validate_input(self, field, *args):
        if field is None:
            logger.error(
                f"Expected event field is missing: {field} [Event -> {''.join(args)}]"
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
        for features, prediction in zip(event["features"], event["prediction"]):
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
        endpoint_id = event["endpoint_id"]

        if endpoint_id not in self.feature_names:
            endpoint_record = get_endpoint_record(
                path=self.kv_path, endpoint_id=endpoint_id,
            )
            feature_names = endpoint_record.get("feature_names")
            feature_names = json.loads(feature_names) if feature_names else None

            if not feature_names:
                logger.warn(
                    f"Seems like endpoint {event['endpoint_id']} was not registered, feature names will be "
                    f"automatically generated"
                )
                feature_names = [f"f{i}" for i, _ in enumerate(event["features"])]
                get_v3io_client().kv.update(
                    container=config.get("CONTAINER"),
                    table_path=self.kv_path,
                    key=event["endpoint_id"],
                    attributes={"feature_names": json.dumps(feature_names)},
                )

            self.feature_names[endpoint_id] = feature_names

        feature_names = self.feature_names[endpoint_id]
        features = event["features"]
        event["named_features"] = {
            name: feature for name, feature in zip(feature_names, features)
        }
        return event


class WriteToKV(MapClass):
    def __init__(self, table: str, **kwargs):
        super().__init__(**kwargs)
        self.table = table

    def do(self, event: Dict):
        get_v3io_client().kv.update(
            container=config.get("CONTAINER"),
            table_path=self.table,
            key=event["endpoint_id"],
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
                token=config.get("V3IO_ACCESS_KEY"),
                container=config.get("CONTAINER"),
                address=config.get("V3IO_FRAMESD"),
            ).execute(backend="kv", table=self.table, command="infer_schema")
        return event


class UpdateParquet(WriteToParquet):
    def __init__(
        self,
        path_template: str,
        index_cols: Union[str, List[str], None] = None,
        columns: Union[str, List[str], None] = None,
        partition_cols: Optional[List[str]] = None,
        infer_columns_from_data: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(
            "", index_cols, columns, partition_cols, infer_columns_from_data, **kwargs
        )

        self.path_template = path_template
        self.projects = set()

    def _makedirs_by_path(self, path):
        fs, file_path = url_to_file_system(path, self._storage_options)
        dirname = os.path.dirname(path)
        if dirname:
            fs.makedirs(dirname, exist_ok=True)

    async def _emit(self, batch, batch_time):
        partition_by_project = defaultdict(list)
        for event in batch:
            project = event[1].split(".")[0]
            partition_by_project[project].append(event)

        for project, events in partition_by_project.items():
            path = self.path_template.format(project=project)
            if project not in self.projects:
                await asyncio.get_running_loop().run_in_executor(
                    None, partial(self._makedirs_by_path, path=path),
                )
            self.projects.add(project)

            df_columns = []
            if self._index_cols:
                df_columns.extend(self._index_cols)
            df_columns.extend(self._columns)
            df = pd.DataFrame(batch, columns=df_columns)
            if self._index_cols:
                df.set_index(self._index_cols, inplace=True)
            df.to_parquet(
                path=path,
                index=bool(self._index_cols),
                partition_cols=self._partition_cols,
                storage_options=self._storage_options,
            )


def _process_before_parquet(batch: List[dict]):
    if batch:
        last_event = batch[-1]["timestamp"]
        for event in batch:
            event["batch_timestamp"] = last_event
            if not event["unpacked_labels"]:
                event["unpacked_labels"] = None
    return batch


def get_endpoint_record(path: str, endpoint_id: str) -> Optional[dict]:
    logger.info(
        f"Grabbing endpoint data", endpoint_id=endpoint_id, table_path=path,
    )
    try:
        endpoint_record = (
            get_v3io_client()
            .kv.get(
                container=config.get("CONTAINER"), table_path=path, key=endpoint_id,
            )
            .output.item
        )
        return endpoint_record
    except Exception:
        return None
