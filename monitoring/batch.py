# from dataclasses import asdict
# from datetime import datetime
from os import environ

# from typing import Dict, Optional, Any, List
#
import pandas as pd

# from mlrun import mount_v3io
# from mlrun.utils import logger
#
# from .clients import get_frames_client
# from .constants import ENDPOINT_DRIFT_LOG
from .drift import VirtualDrift
# from .endpoint import EndpointKey
from v3iofs import V3ioFS
import operator


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

        # mount_v3io(
        #     remote="~/monitoring",
        #     mount_path="/v3io",
        #     access_key=environ.get("V3IO_ACCESS_KEY"),
        # )
        #
        # self.virtual_drift = VirtualDrift(
        #     bin_count, feature_cols, prediction_col, label_col
        # )
        # logger.info("Creating virtual drift table...")
        # get_frames_client().create(
        #     backend="tsdb", table=ENDPOINT_DRIFT_LOG, rate="60/h", if_exists=1
        # )
        # logger.info("Done creating virtual drift table.")

    # def process(self, endpoint_key: EndpointKey, base_dataset: pd.DataFrame):
    #     other_dataset = self.get_last_parquet_log(endpoint_key)
    #     drift_measures = self.virtual_drift.compute_drift(base_dataset, other_dataset)
    #
    #     drift_measures.update(
    #         timestamp=pd.to_datetime(str(datetime.now())), **asdict(endpoint_key)
    #     )
    #
    #     self.check_drift_alerts(drift_measures)
    #
    #     results_df = pd.DataFrame([drift_measures])
    #     results_df = results_df.set_index(["timestamp", "hash"])
    #     get_frames_client().write("tsdb", ENDPOINT_DRIFT_LOG, dfs=results_df)
    #
    # def check_drift_alerts(self, results: Dict[str, Any]):
    #     pass
    #
    # def get_last_parquet_log(self, endpoint_key: EndpointKey) -> pd.DataFrame:
    #     pass


if __name__ == "__main__":
    EventBatchProcessor()
