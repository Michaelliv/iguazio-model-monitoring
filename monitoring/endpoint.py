from dataclasses import dataclass
from typing import Optional


@dataclass()
class EndpointKey:
    project: str
    function: str
    model: str
    tag: str
    model_class: Optional[str] = None
    hash: Optional[str] = None

    def __post_init__(self):
        self.hash: str = f"{self.project}_{self.function}_{self.model}_{self.tag}"

    def __str__(self):
        return self.hash
