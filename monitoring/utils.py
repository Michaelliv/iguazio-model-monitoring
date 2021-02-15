from typing import Dict, Any
import hashlib


def endpoint_details_from_event(request: Dict) -> Dict[str, Any]:
    project, function_with_tag = request["function_uri"].split("/")

    try:
        function, tag = function_with_tag.split(":")
    except ValueError:
        function, tag = function_with_tag, "latest"

    model = request["model"]
    model_class = request.get("class")

    return {
        "project": project,
        "model": model,
        "function": function,
        "tag": tag,
        "model_class": model_class,
        "labels": request.get("labels"),
    }


def endpoint_id_from_details(details: Dict) -> str:
    endpoint_unique_string = f"{details['function']}_{details['model']}_{details['tag']}"
    md5 = hashlib.md5(endpoint_unique_string.encode('utf-8')).hexdigest()
    return f"{details['project']}.{md5}"
