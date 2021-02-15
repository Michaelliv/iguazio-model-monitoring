from os import environ

import pandas as pd
from v3iofs import V3ioFS

from .drift import VirtualDrift


class EventBatchProcessor:
    def __init__(
        self,
        # bin_count: int,
        # feature_cols: Optional[List[str]],
        # prediction_col: Optional[str],
        # label_col: Optional[str],
    ):
        self.v3io_fs = V3ioFS(
            v3io_api=environ.get("V3IO_API"),
            v3io_access_key=environ.get("V3IO_ACCESS_KEY"),
        )

    def run(self):
        for batch_path in self._iter_latest_data():
            batch = pd.read_parquet(batch_path)
            VirtualDrift()

    def _iter_latest_data(self):
        try:
            endpoint_directories = self.v3io_fs.listdir("monitoring/event_batch")
            for endpoint_directory in endpoint_directories:
                endpoint_batches = self.v3io_fs.listdir(endpoint_directory["name"])
                last_created = max(endpoint_batches, key=lambda d: d["mtime"])
                data = self.v3io_fs.listdir(last_created["name"])
                yield f"v3io://{data[0]['name']}"
        except FileNotFoundError:
            return None


if __name__ == "__main__":
    EventBatchProcessor()
