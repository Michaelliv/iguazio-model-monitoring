import json
from datetime import datetime
from os import environ
from typing import Dict, List, Any, Set, Tuple, Optional
from mlrun import mount_v3io

from pandas import to_datetime
from storey import (
    FieldAggregator,
    NoopDriver,
    Table,
    Source,
    Map,
    MapWithState,
    AggregateByKey,
    build_flow,
    FlatMap,
    WriteToTSDB,
    WriteToParquet,
)
from storey.dtypes import SlidingWindows
from storey.steps import ForEach, Sample, Partition

from model_monitoring.clients import get_v3io_client
from model_monitoring.constants import ISO_8601
from model_monitoring.utils import (
    endpoint_details_from_event,
    endpoint_id_from_details,
)


class ProcessorState:
    def __init__(self):
        self.active_endpoints: Set[str] = set()
        self.first_request: Dict[str, str] = dict()


def _pass(event):
    pass


class EventStreamProcessor:
    def __init__(self):
        mount_v3io(
            remote="~/monitoring",
            mount_path="/v3io",
            access_key=environ.get("V3IO_ACCESS_KEY")
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
            "named_features",
            "latency_avg_1s",
            "predictions_per_second_count_1s",
            "first_request",
        ]

        self._tsdb_keys = [
            "timestamp",
            "project",
            "model",
            "function",
            "tag",
            "model_class",
            "endpoint_id",
            "named_features",
            "prediction",
            "predictions_per_second_count_1s",
            "latency_avg_1s",
        ]

        self._state = ProcessorState()

        self._flow = build_flow(
            [
                Source(),
                MapWithState(self._state, _process_endpoint_event),
                FlatMap(unpack_predictions),
                # Branch 1: Aggregate events, count averages and update TSDB and KV
                # [
                #     AggregateByKey(
                #         aggregates=[
                #             FieldAggregator(
                #                 "predictions_per_second",
                #                 "endpoint_id",
                #                 ["count"],
                #                 SlidingWindows(["1s"], "1s"),
                #             ),
                #             FieldAggregator(
                #                 "latency",
                #                 "latency",
                #                 ["avg"],
                #                 SlidingWindows(["1s"], "1s"),
                #             ),
                #         ],
                #         table=Table("notable", NoopDriver()),
                #     ),
                #     Sample(10),
                #     # Branch 1.1: Updated KV
                #     [
                #         Map(self.process_before_kv),
                #         Map(
                #             lambda e: get_v3io_client().kv.update(
                #                 container="monitoring",
                #                 table_path="endpoints",
                #                 key=e["endpoint_id"],
                #                 attributes=e,
                #             )
                #         ),
                #     ],
                #     # Branch 1.2: Update TSDB
                #     [
                #         Map(self.process_before_tsdb),
                #         ForEach(_pass),
                #         WriteToTSDB(
                #             path="endpoint_events",
                #             time_col="timestamp",
                #             infer_columns_from_data=True,
                #             index_cols=["endpoint_id"],
                #             v3io_frames=environ.get("V3IO_FRAMES"),
                #             access_key=environ.get("V3IO_ACCESS_KEY"),
                #             container="monitoring",
                #             rate="1/s",
                #             max_events=100,  # Every 100 sampled events or
                #             timeout_secs=60 * 5  # Every 5 minutes
                #         ),
                #     ],
                # ],
                # Branch 2: Batch events, write to parquet
                [
                    WriteToParquet(
                        path="v3io/event_batch",
                        partition_cols=["endpoint_id", "timestamp"],
                        # Settings for batching
                        max_events=100,  # Every 1000 events or
                        timeout_secs=60 * 5,  # Every 5 minutes
                        key="endpoint_id"
                    ),
                ],
            ]
        ).run()

    def process_before_kv(self, event: Dict):
        e = {k: event[k] for k in self._tsdb_keys}
        e = {**e, **e.pop("unpacked_labels", {})}
        return e

    def process_before_tsdb(self, event: Dict):
        e = {k: event[k] for k in self._tsdb_keys}
        e = {**e, **e.pop("named_features", {})}
        e["timestamp"] = to_datetime(e["timestamp"], format=ISO_8601)
        e["labels"] = json.dumps(e["labels"])
        e["named_features"] = json.dumps(e["named_features"])
        return e


def unpack_labels(labels: Optional[List[str]]) -> Dict[str, str]:
    if not labels:
        return {}

    unpacked = {}
    for label in labels:
        lbl, value = label.split("==")
        unpacked[f"_{lbl}"] = value

    return unpacked


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


def write_to_kv(event: Dict):
    get_v3io_client().kv.update(
        container="monitoring",
        table_path="endpoints",
        key=event["endpoint_id"],
        attributes=event,
    )


def _process_endpoint_event(
    event: Dict, state: ProcessorState
) -> Tuple[Dict, ProcessorState]:

    endpoint_details = endpoint_details_from_event(event)
    endpoint_id = endpoint_id_from_details(endpoint_details)

    if endpoint_id not in state.active_endpoints:
        state.active_endpoints.add(endpoint_id)
        state.first_request[endpoint_id] = event["when"]

    unpacked_labels = unpack_labels(event["labels"])

    event = {
        "timestamp": event["when"],
        "endpoint_id": endpoint_id,
        "request_id": event["request"]["id"],
        "latency": event["microsec"],
        "features": event["request"]["resp"]["outputs"]["inputs"],
        "prediction": event["request"]["resp"]["outputs"]["prediction"],
        "first_request": state.first_request[endpoint_id],
        "unpacked_labels": unpacked_labels,
        **endpoint_details,
    }

    return event, state


if __name__ == "__main__":
    from time import sleep
    from random import randint, choice, uniform
    from uuid import uuid1

    import string
    from sklearn.datasets import load_iris

    iris = load_iris()
    iris_data = iris["data"].tolist()
    iris_target = iris["target"].tolist()

    def get_random_event(model_details):
        random_indexes = [randint(0, len(iris_data) - 1) for _ in range(randint(1, 5))]
        data = [iris_data[i] for i in random_indexes]
        targets = [iris_target[i] for i in random_indexes]

        return {
            "class": model_details["class"],
            "model": model_details["model"],
            "labels": model_details["labels"],
            "function_uri": f"{model_details['project']}/{model_details['function']}:{model_details['tag']}",
            "when": str(datetime.now()),
            "microsec": randint(10_000, 50_000),
            "request": {
                "id": str(uuid1()),
                "resp": {"outputs": {"inputs": data, "prediction": targets}},
            },
        }

    def get_random_endpoint_details() -> Dict[str, Any]:
        return {
            "project": "test",
            "model": f"model_{randint(0, 100)}",
            "function": f"function_{randint(0, 100)}",
            "tag": f"v{randint(0, 100)}",
            "class": "classifier",
            "labels": [
                f"{choice(string.ascii_letters)}=={randint(0, 100)}"
                for _ in range(1, 5)
            ],
        }

    esp = EventStreamProcessor()

    endpoints = [get_random_endpoint_details() for i in range(0, 5)]

    while True:
        for i, endpoint in enumerate(endpoints):
            event = get_random_event(endpoint)
            esp._flow.emit(event)
        sleep(uniform(0.3, 0.6))
