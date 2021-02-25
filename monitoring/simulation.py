import string
from datetime import datetime
from random import randint, choice, uniform
from time import sleep
from typing import Any, Dict
from uuid import uuid1

import pandas as pd
from sklearn.datasets import load_iris

from monitoring.stream import EventStreamProcessor


def run_shiba_simulation():

    shiba = pd.read_parquet("tests/sample.pq")
    shiba = shiba.to_dict(orient="records")

    def get_random_event_shiba(model_details):
        random_indexes = [randint(0, len(shiba) - 1) for _ in range(randint(1, 5))]
        data = [list(shiba[i].values()) for i in random_indexes]
        targets = [uniform(0, 1) for _ in random_indexes]

        event = {
            "class": model_details["class"],
            "model": model_details["model"],
            "labels": model_details["labels"],
            "function_uri": f"{model_details['project']}/{model_details['function']}:{model_details['tag']}",
            "when": str(datetime.utcnow()),
            "microsec": randint(10_000, 50_000),
            "request": {"id": str(uuid1()), "inputs": data,},
            "resp": {"outputs": targets},
        }

        return event

    def generate_endpoints_shiba():
        return [
            {
                "project": "test",
                "model": "patient_det_lr",
                "function": "patient-deterioration",
                "tag": f"v{randint(0, 100)}",
                "class": "classifier",
                "labels": {
                    f"{choice(string.ascii_letters)}": randint(0, 100)
                    for _ in range(1, 2)
                },
            },
            {
                "project": "test",
                "model": "patient_det_rf",
                "function": "patient-deterioration",
                "tag": f"v{randint(0, 100)}",
                "class": "classifier",
                "labels": {
                    f"{choice(string.ascii_letters)}": randint(0, 100)
                    for _ in range(1, 2)
                },
            },
            {
                "project": "test",
                "model": f"patient_det_adaboost",
                "function": "patient-deterioration",
                "tag": f"v{randint(0, 100)}",
                "class": "classifier",
                "labels": {
                    f"{choice(string.ascii_letters)}": randint(0, 100)
                    for _ in range(1, 2)
                },
            },
            {
                "project": "test",
                "model": "VotingEnsemble",
                "function": "patient-deterioration",
                "tag": f"v{randint(0, 100)}",
                "class": "classifier",
                "labels": {
                    f"{choice(string.ascii_letters)}": randint(0, 100)
                    for _ in range(1, 2)
                },
            },
        ]

    esp = EventStreamProcessor("")

    # endpoints = [get_random_endpoint_details() for i in range(0, 5)]
    endpoints = generate_endpoints_shiba()

    while True:
        for i, endpoint in enumerate(endpoints):
            event = get_random_event_shiba(endpoint)
            esp.consume(event)
        sleep(uniform(0.6, 1.2))


def run_test_simulation():

    iris = load_iris()
    iris_data = iris["data"].tolist()

    def get_random_event(model_details):
        random_indexes = [randint(0, len(iris_data) - 1) for _ in range(randint(1, 5))]
        data = [list(iris_data[i]) for i in random_indexes]
        targets = [uniform(0, 1) for _ in random_indexes]

        event = {
            "class": model_details["class"],
            "model": model_details["model"],
            "labels": model_details["labels"],
            "function_uri": f"{model_details['project']}/{model_details['function']}:{model_details['tag']}",
            "when": str(datetime.utcnow()),
            "microsec": randint(10_000, 50_000),
            "request": {"id": str(uuid1()), "inputs": data,},
            "resp": {"outputs": targets},
        }
        return event

    def get_random_endpoint_details() -> Dict[str, Any]:
        return {
            "project": "test",
            "model": f"model_{randint(0, 100)}",
            "function": f"function_{randint(0, 100)}",
            "tag": f"v{randint(0, 100)}",
            "class": "classifier",
            "labels": {
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            },
        }

    esp = EventStreamProcessor("{project}")

    endpoints = [get_random_endpoint_details() for i in range(0, 5)]

    while True:
        for i, endpoint in enumerate(endpoints):
            event = get_random_event(endpoint)
            esp.consume(event)
        sleep(uniform(0.6, 1.2))



def run_full_iris_simulation():

    iris = load_iris()
    iris_data = iris["data"].tolist()

    def get_random_event(model_details):
        targets = [uniform(0, 1) for _ in iris_data]

        event = {
            "class": model_details["class"],
            "model": model_details["model"],
            "labels": model_details["labels"],
            "function_uri": f"{model_details['project']}/{model_details['function']}:{model_details['tag']}",
            "when": str(datetime.utcnow()),
            "microsec": randint(10_000, 50_000),
            "request": {"id": str(uuid1()), "inputs": iris_data},
            "resp": {"outputs": targets},
        }
        return event

    def get_random_endpoint_details() -> Dict[str, Any]:
        return {
            "project": "test",
            "model": f"model_{randint(0, 100)}",
            "function": f"function_{randint(0, 100)}",
            "tag": f"v{randint(0, 100)}",
            "class": "classifier",
            "labels": {
                f"{choice(string.ascii_letters)}": randint(0, 100) for _ in range(1, 5)
            },
        }

    esp = EventStreamProcessor("{project}")

    endpoints = [get_random_endpoint_details() for i in range(0, 5)]

    while True:
        for i, endpoint in enumerate(endpoints):
            event = get_random_event(endpoint)
            esp.consume(event)
        sleep(uniform(0.6, 1.2))


if __name__ == "__main__":
    run_full_iris_simulation()
