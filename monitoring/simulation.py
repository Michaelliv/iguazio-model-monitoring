import string
from datetime import datetime
from random import randint, choice, uniform, choices
from time import sleep
from typing import Any, Dict
from uuid import uuid1

from sklearn.datasets import load_iris

from monitoring.stream import EventStreamProcessor

if __name__ == "__main__":

    iris = load_iris()
    iris_data = iris["data"].tolist()
    iris_target = iris["target"].tolist()

    def get_random_event(model_details):
        random_indexes = [randint(0, len(iris_data) - 1) for _ in range(randint(1, 5))]
        data = [iris_data[i] for i in random_indexes]
        targets = [iris_target[i] for i in random_indexes]

        is_error = choices([True, False], [0.005, 0.995])

        event = {
            "class": model_details["class"],
            "model": model_details["model"],
            "labels": model_details["labels"],
            "function_uri": f"{model_details['project']}/{model_details['function']}:{model_details['tag']}",
            "when": str(datetime.utcnow()),
            "microsec": randint(10_000, 50_000),
            "request": {
                "id": str(uuid1()),
                "resp": {"outputs": {"inputs": data, "prediction": targets}},
            },
        }

        if is_error:
            event["error"] = "Simulated Error"

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

    esp = EventStreamProcessor()

    endpoints = [get_random_endpoint_details() for i in range(0, 5)]

    while True:
        for i, endpoint in enumerate(endpoints):
            event = get_random_event(endpoint)
            esp._flow.emit(event)
        sleep(uniform(0.6, 1.2))
