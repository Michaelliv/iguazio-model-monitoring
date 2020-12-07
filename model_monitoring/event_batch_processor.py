from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Any, List

import pandas as pd
from mlrun.utils import logger

from model_monitoring.clients import get_frames_client
from model_monitoring.constants import ENDPOINT_DRIFT_LOG
from model_monitoring.drift import VirtualDrift
from model_monitoring.endpoint import EndpointKey


class EventBatchProcessor:
    def __init__(
        self,
        bin_count: int,
        feature_cols: Optional[List[str]],
        prediction_col: Optional[str],
        label_col: Optional[str],
    ):
        self.virtual_drift = VirtualDrift(
            bin_count, feature_cols, prediction_col, label_col
        )
        logger.info("Creating virtual drift table...")
        get_frames_client().create(
            backend="tsdb", table=ENDPOINT_DRIFT_LOG, rate="60/h", if_exists=1
        )
        logger.info("Done creating virtual drift table.")

    def process(self, endpoint_key: EndpointKey, base_dataset: pd.DataFrame):
        other_dataset = self.get_last_parquet_log(endpoint_key)
        drift_measures = self.virtual_drift.compute_drift(base_dataset, other_dataset)

        drift_measures.update(
            timestamp=pd.to_datetime(str(datetime.now())), **asdict(endpoint_key)
        )

        self.check_drift_alerts(drift_measures)

        results_df = pd.DataFrame([drift_measures])
        results_df = results_df.set_index(["timestamp", "hash"])
        get_frames_client().write("tsdb", ENDPOINT_DRIFT_LOG, dfs=results_df)

    def check_drift_alerts(self, results: Dict[str, Any]):
        pass

    def get_last_parquet_log(self, endpoint_key: EndpointKey) -> pd.DataFrame:
        pass
