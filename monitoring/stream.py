import asyncio
import json
import os
from collections import defaultdict
from functools import partial
from os import environ
from typing import Dict, List, Set, Optional, Callable, Union

import pandas as pd
import v3io_frames
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
)
from storey.dtypes import SlidingWindows
from storey.steps import SampleWindow
from storey.utils import url_to_file_system

from .clients import get_v3io_client, get_frames_client
from .constants import ISO_8601
from .utils import (
    endpoint_details_from_event,
    endpoint_id_from_details,
)


class EventStreamProcessor:
    def __init__(self, parquet_path_template: str):
        self.parquet_path_template = parquet_path_template

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

        self._events_tsdb_keys = [
            "timestamp",
            "project",
            "model",
            "function",
            "tag",
            "model_class",
            "endpoint_id",
            "predictions_per_second_count_1s",
            "latency_avg_1s",
        ]

        self._features_tsdb_keys = [
            "timestamp",
            "endpoint_id",
            "project",
            "named_features",
            "prediction",
        ]

        self._flow = build_flow(
            [
                Source(),
                ProcessEndpointEvent(),
                FlatMap(self.unpack_predictions),
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
                    SampleWindow(10),
                    # Branch 1.1: Updated KV
                    [
                        Map(self.process_before_kv),
                        UpdateKV(lambda e: f"{e['project']}/model-endpoints"),
                        InferSchema(lambda e: f"{e['project']}/model-endpoints"),
                    ],
                    # Branch 1.2: Update TSDB
                    [
                        Map(self.process_before_events_tsdb),
                        Batch(max_events=100, timeout_secs=60 * 5),
                        UpdateTSDB(
                            path_builder=lambda e: f"{e[-1]['project']}/endpoint-events",
                            tsdb_columns=self._events_tsdb_keys,
                            rate="10/m",
                        ),
                    ],
                    [
                        Map(self.process_before_features_tsdb),
                        Batch(
                            max_events=10,
                            timeout_secs=60 * 5,
                            key=lambda e: e.body["endpoint_id"],
                        ),
                        UpdateTSDB(
                            path_builder=lambda e: f"{e[-1]['project']}/endpoint-features",
                            rate="10/m",
                            infer_columns=True,
                            exclude_columns={"project"},
                        ),
                    ],
                ],
                # Branch 2: Batch events, write to parquet
                [
                    Batch(
                        max_events=10,  # Every 1000 events or
                        timeout_secs=60 * 5,  # Every 5 minutes
                        key="endpoint_id",
                    ),
                    FlatMap(lambda batch: _mark_batch_timestamp(batch)),
                    UpdateParquet(
                        path_template=self.parquet_path_template,
                        partition_cols=["endpoint_id", "batch_timestamp"],
                        # Settings for _Batching
                        max_events=10,  # Every 1000 events or
                        timeout_secs=60 * 5,  # Every 5 minutes
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
            predictions.append(
                dict(
                    event,
                    features=features,
                    prediction=prediction,
                    named_features={f"f{i}": value for i, value in enumerate(features)},
                )
            )
        return predictions

    def process_before_kv(self, event: Dict):
        e = {k: event[k] for k in self._kv_keys}
        e = {**e, **e.pop("unpacked_labels", {})}
        e["labels"] = json.dumps(e["labels"])
        return e

    def process_before_events_tsdb(self, event: Dict):
        e = {k: event[k] for k in self._events_tsdb_keys}
        e["timestamp"] = pd.to_datetime(e["timestamp"], format=ISO_8601, utc=True)
        return e

    def process_before_features_tsdb(self, event: Dict):
        e = {k: event[k] for k in self._features_tsdb_keys}
        e = {**e, **e.pop("named_features", {})}
        e["timestamp"] = pd.to_datetime(e["timestamp"], format=ISO_8601, utc=True)
        return e


class ProcessEndpointEvent(MapClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_request: Dict[str, str] = dict()
        self.last_request: Dict[str, str] = dict()
        self.error_count: Dict[str, int] = defaultdict(int)

    def do(self, event: dict):
        endpoint_details = endpoint_details_from_event(event)
        endpoint_id = endpoint_id_from_details(endpoint_details)
        when = event["when"]

        # error = event.get("error")
        # if error:
        #     try:
        #         self.error_count[endpoint_id] += 1
        #         client = get_frames_client(
        #             token=environ.get("V3IO_ACCESS_KEY"),
        #             container="projects",
        #             address=environ.get("V3IO_FRAMESD"),
        #         )
        #
        #         ts_error = {
        #             "timestamp": pd.to_datetime(when, format=ISO_8601, utc=True),
        #             "endpoint_id": endpoint_id,
        #             "error": error,
        #         }
        #
        #         df = pd.DataFrame([ts_error])
        #         df.set_index(["timestamp", "endpoint_id"])
        #         path = f"{endpoint_details['project']}/error_log"
        #         if path not in self.error_paths:
        #             self.error_paths.add(path)
        #             client.create(
        #                 backend="tsdb",
        #                 table=path,
        #                 if_exists=v3io_frames.frames_pb2.IGNORE,
        #                 rate="10/m",
        #             )
        #         client.write("tsdb", path, df)
        #
        #     except Exception as e:
        #         print(e)
        # else:

        if endpoint_id not in self.first_request:
            self.first_request[endpoint_id] = when

        self.last_request[endpoint_id] = when

        event = {
            "timestamp": when,
            "endpoint_id": endpoint_id,
            "request_id": event["request"]["id"],
            "latency": event["microsec"],
            "features": event["request"]["inputs"],
            "prediction": event["resp"]["outputs"],
            "first_request": self.first_request[endpoint_id],
            "last_request": self.last_request[endpoint_id],
            "error_count": self.error_count[endpoint_id],
            "unpacked_labels": {f"_{k}": v for k, v in event.get("labels", {}).items()},
            **endpoint_details,
        }

        return event


class UpdateKV(MapClass):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path_getter = path

    def do(self, event: Dict):
        path = self.path_getter(event)
        get_v3io_client().kv.update(
            container="projects",
            table_path=path,
            key=event["endpoint_id"],
            attributes=event,
        )
        return event


class UpdateTSDB(MapClass):
    def __init__(
        self,
        path_builder: Callable[[List[dict]], str],
        rate: str,
        tsdb_columns: Optional[List[str]] = None,
        exclude_columns: Optional[Set[str]] = None,
        infer_columns: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path_builder = path_builder
        self.rate = rate
        self.tsdb_columns = tsdb_columns or []
        self.exclude_columns = exclude_columns or set()
        self.infer_columns = infer_columns
        self.projects = set()

    def do(self, event: List[dict]):
        if not event:
            return

        path = self.path_builder(event)

        client = get_frames_client(
            token=environ.get("V3IO_ACCESS_KEY"),
            container="projects",
            address=environ.get("V3IO_FRAMESD"),
        )

        if self.exclude_columns:
            if self.tsdb_columns:
                self.tsdb_columns = [
                    c for c in self.tsdb_columns if c not in self.exclude_columns
                ]
            for e in event:
                for exclude in self.exclude_columns:
                    e.pop(exclude, None)

        if self.tsdb_columns and not self.infer_columns:
            columns = self.tsdb_columns
        elif self.infer_columns and not self.tsdb_columns:
            columns = list(event[0].keys())
        else:
            raise RuntimeError("Failed to get tsdb columns")

        df = pd.DataFrame(event, columns=columns)
        df.set_index(keys=["timestamp", "endpoint_id"], inplace=True)

        if path not in self.projects:
            self.projects.add(path)

            client.create(
                backend="tsdb",
                table=path,
                if_exists=v3io_frames.frames_pb2.IGNORE,
                rate=self.rate,
            )

        client.write("tsdb", path, df)


class InferSchema(MapClass):
    def __init__(self, key_generator, **kwargs):
        super().__init__(**kwargs)
        self.key_generator = key_generator
        self.inferred = {}

    def do(self, event: Dict):
        key = self.key_generator(event)
        key_set = set(event.keys())
        if key not in self.inferred:
            self.inferred[key] = key_set
            get_frames_client(
                token=environ.get("V3IO_ACCESS_KEY"),
                container="projects",
                address=environ.get("V3IO_FRAMESD"),
            ).execute(backend="kv", table=key, command="infer_schema")
        else:
            if not key_set.issubset(self.inferred[key]):
                self.inferred[key] = self.inferred[key].union(key_set)
                get_frames_client(
                    token=environ.get("V3IO_ACCESS_KEY"),
                    container="projects",
                    address=environ.get("V3IO_FRAMESD"),
                ).execute(backend="kv", table=key, command="infer_schema")
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


def _mark_batch_timestamp(batch: List[dict]):
    if batch:
        last_event = batch[-1]["timestamp"]
        for event in batch:
            event["batch_timestamp"] = last_event
    return batch
