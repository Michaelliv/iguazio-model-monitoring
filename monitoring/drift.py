from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from mlrun.data_types.infer import DFDataInfer, InferOptions
from sklearn.preprocessing import KBinsDiscretizer

from monitoring.measurements import (
    TotalVarianceDistance,
    HellingerDistance,
    KullbackLeiblerDivergence,
)


class VirtualDrift:
    def __init__(
        self,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
        feature_weights: Optional[List[float]] = None,
        inf_capping: Optional[float] = 10,
        kld_zero_scaling: Optional[float] = 0.001,
    ):
        self.prediction_col = prediction_col
        self.label_col = label_col
        self.feature_weights = feature_weights
        self.capping = inf_capping
        self.discretizers: Dict[str, KBinsDiscretizer] = {}
        self.metrics = {
            "TVD": TotalVarianceDistance,
            "Hellinger": HellingerDistance,
            "KLD": KullbackLeiblerDivergence,
        }

    def yaml_to_histogram(self, histogram_yaml):
        histograms = {
            feature: feature_stats["hist"][0]
            for feature, feature_stats in histogram_yaml["feature_stats"].items()
        }
        # Get features value counts
        histograms = pd.concat(
            [
                pd.DataFrame(data=hist, columns=[feature])
                for feature, hist in histograms.items()
            ],
            axis=1,
        )
        # To Distribution
        histograms = histograms / histograms.sum()
        return histograms

    def compute_metrics_over_df(self, base_histogram, latest_histogram):
        drift_measures = {}
        for metric_name, metric in self.metrics.items():
            drift_measures[metric_name] = {
                feature: metric(
                    base_histogram.loc[:, feature], latest_histogram.loc[:, feature]
                ).compute()
                for feature in base_histogram
            }
        return drift_measures

    def compute_drift_from_histogram_and_df(self, base_histogram, latest_parquet_path):
        df = pd.read_parquet(latest_parquet_path)
        df = list(df["named_features"])
        df = pd.DataFrame(df)
        latest_stats = DFDataInfer.get_stats(df, InferOptions.Histogram)
        latest_stats = {"feature_stats": latest_stats}

        result_drift = self.compute_drift_from_histograms(base_histogram, latest_stats)
        return result_drift

    def compute_drift_from_histograms(self, base_histogram_yaml, latest_histogram_yaml):

        # Process histogram yamls to Dataframe of the histograms
        # with Feature histogram as cols
        base_histogram = self.yaml_to_histogram(base_histogram_yaml)
        latest_histogram = self.yaml_to_histogram(latest_histogram_yaml)

        # Verify all the features exist between datasets
        base_features = set(base_histogram.columns)
        latest_features = set(latest_histogram.columns)
        if not base_features == latest_features:
            raise ValueError(
                f"Base dataset and latest dataset have different featuers: {base_features} <> {latest_features}"
            )

        # Compute the drift per feature
        features_drift_measures = self.compute_metrics_over_df(
            base_histogram.loc[:, base_features], latest_histogram.loc[:, base_features]
        )

        # Compute total drift measures for features
        for metric_name in self.metrics.keys():
            feature_values = list(features_drift_measures[metric_name].values())
            features_drift_measures[metric_name]["total_sum"] = np.sum(feature_values)
            features_drift_measures[metric_name]["total_mean"] = np.mean(feature_values)

            # Add weighted mean by given feature weights if provided
            if self.feature_weights:
                features_drift_measures[metric_name]["total_weighted_mean"] = np.dot(
                    feature_values, self.feature_weights
                )

        result_drift = {
            f"{feature}_{metric}": value
            for metric in features_drift_measures.keys()
            for feature, value in features_drift_measures[metric].items()
        }

        if self.label_col:
            label_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.label_col],
                latest_histogram.loc[:, self.label_col],
            )
            result_drift.update(
                {
                    f"{self.label_col}_{metric}": label_drift_measures[metric][
                        self.label_col
                    ]
                    for metric in label_drift_measures.keys()
                }
            )
        if self.prediction_col:
            prediction_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.prediction_col],
                latest_histogram.loc[:, self.prediction_col],
            )
            result_drift.update(
                {
                    f"{self.prediction_col}_{metric}": prediction_drift_measures[
                        metric
                    ][self.prediction_col]
                    for metric in prediction_drift_measures.keys()
                }
            )

        return result_drift
