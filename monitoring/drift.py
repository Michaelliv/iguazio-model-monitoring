from collections import defaultdict
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
            "tvd": TotalVarianceDistance,
            "hellinger": HellingerDistance,
            "kld": KullbackLeiblerDivergence,
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

        drift_result = defaultdict(dict)

        for feature in base_features:
            for metric, values in features_drift_measures.items():
                drift_result[feature][metric] = values[feature]
                sum = features_drift_measures[metric]["total_sum"]
                mean = features_drift_measures[metric]["total_mean"]
                drift_result[f"{metric}_sum"] = sum
                drift_result[f"{metric}_mean"] = mean
                if self.feature_weights:
                    metric_measure = features_drift_measures[metric]
                    weighted_mean = metric_measure["total_weighted_mean"]
                    drift_result[f"{metric}_weighted_mean"] = weighted_mean

        if self.label_col:
            label_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.label_col],
                latest_histogram.loc[:, self.label_col],
            )
            for metric, values in label_drift_measures.items():
                drift_result[self.label_col][metric] = values[metric]

        if self.prediction_col:
            prediction_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.prediction_col],
                latest_histogram.loc[:, self.prediction_col],
            )
            for metric, values in prediction_drift_measures.items():
                drift_result[self.prediction_col][metric] = values[metric]

        return drift_result

    @staticmethod
    def parquet_to_stats(parquet_path: str):
        df = pd.read_parquet(parquet_path)
        df = list(df["named_features"])
        df = pd.DataFrame(df)
        latest_stats = DFDataInfer.get_stats(df, InferOptions.Histogram)
        latest_stats = {"feature_stats": latest_stats}
        return latest_stats
