from typing import Optional, List, Dict, Tuple

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from src.drift.measurements import (
    TotalVarianceDistance,
    HellingerDistance,
    KullbackLeiblerDivergence,
)


class VirtualDrift:
    def __init__(
        self,
        bin_count: int,
        feature_cols: Optional[List[str]],
        prediction_col: Optional[str],
        label_col: Optional[str],
    ):
        self.feature_cols = feature_cols
        self.prediction_col = prediction_col
        self.label_col = label_col
        self.bin_count = bin_count
        self.discretizers: Dict[str, KBinsDiscretizer] = {}

    def compute_drift(
        self,
        first_dataset: pd.DataFrame,
        second_dataset: pd.DataFrame,
    ) -> Dict[str, float]:
        self.init_discritizers(first_dataset)
        self.discritize(first_dataset, second_dataset)
        return self.compute_drift_measures(first_dataset, second_dataset)

    def init_discritizers(self, base_data):
        continuous_features = base_data.select_dtypes(["float"])
        for feature in continuous_features.columns:
            if feature not in self.discretizers:
                self.discretizers[feature] = self.fit_discretizer(
                    feature, continuous_features
                )

    def fit_discretizer(
        self, feature_name: str, continuous_features: pd.DataFrame
    ) -> KBinsDiscretizer:
        discretizer = KBinsDiscretizer(
            n_bins=self.bin_count, encode="ordinal", strategy="uniform"
        )
        discretizer.fit(continuous_features.loc[:, feature_name].values.reshape(-1, 1))
        return discretizer

    def discritize(self, base_data, current_data):
        for feature, discretizer in self.discretizers.items():
            base_data[feature] = discretizer.transform(
                base_data.loc[:, feature].values.reshape(-1, 1)
            )
            current_data[feature] = discretizer.transform(
                current_data.loc[:, feature].values.reshape(-1, 1)
            )
            base_data[feature] = base_data[feature].astype("int")
            current_data[feature] = current_data[feature].astype("int")

    def compute_drift_measures(
        self,
        base_data,
        current_data,
    ) -> Dict[str, float]:
        feature_drift = self.compute_feature_drift(base_data, current_data)
        prediction_drift = self.compute_prediction_drift(base_data, current_data)
        label_drift = self.compute_label_drift(base_data, current_data)

        drift_measures = {**feature_drift, **label_drift, **prediction_drift}

        # Normalize inf values
        for key, value in drift_measures.items():
            if value == float("inf"):
                drift_measures[key] = 10

        return drift_measures

    def compute_feature_drift(
        self,
        base_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> Dict[str, float]:
        if self.feature_cols is None:
            return {}

        base_features = base_data[self.feature_cols]
        current_features = current_data[self.feature_cols]
        features_t, features_u = self.to_observations(base_features, current_features)

        return {
            "features_tvd": TotalVarianceDistance(features_t, features_u).compute(),
            "features_hd": HellingerDistance(features_t, features_u).compute(),
            "features_kld": KullbackLeiblerDivergence(features_t, features_u).compute(),
        }

    def compute_prediction_drift(
        self,
        base_data: pd.DataFrame,
        current_data: pd.DataFrame,
    ) -> Dict[str, float]:
        if self.prediction_col is None:
            return {}

        base_predictions = pd.DataFrame(base_data.loc[:, self.prediction_col])
        current_predictions = pd.DataFrame(current_data.loc[:, self.prediction_col])
        prediction_t, prediction_u = self.to_observations(
            base_predictions, current_predictions
        )

        return {
            "prediction_tvd": TotalVarianceDistance(
                prediction_t, prediction_u
            ).compute(),
            "prediction_hd": HellingerDistance(prediction_t, prediction_u).compute(),
            "prediction_kld": KullbackLeiblerDivergence(
                prediction_t, prediction_u
            ).compute(),
        }

    def compute_label_drift(
        self, base_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> Dict[str, float]:
        if self.label_col is None:
            return {}

        base_labels = pd.DataFrame(base_data.loc[:, self.label_col])
        current_labels = pd.DataFrame(current_data.loc[:, self.label_col])
        labels_t, labels_u = self.to_observations(base_labels, current_labels)

        return {
            "prediction_tvd": TotalVarianceDistance(labels_t, labels_u).compute(),
            "prediction_hd": HellingerDistance(labels_t, labels_u).compute(),
            "prediction_kld": KullbackLeiblerDivergence(labels_t, labels_u).compute(),
        }

    @staticmethod
    def to_observations(
        t: pd.DataFrame, u: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series]:
        t = (
            t.apply(
                lambda row: f"{'_'.join([str(row[val]) for val in t.columns])}", axis=1
            )
            .value_counts()
            .sort_index()
        )
        u = (
            u.apply(
                lambda row: f"{'_'.join([str(row[val]) for val in u.columns])}", axis=1
            )
            .value_counts()
            .sort_index()
        )

        joined_uniques = pd.DataFrame([t, u]).T.fillna(0).sort_index()
        joined_uniques.columns = ["t", "u"]

        t_obs = joined_uniques.loc[:, "t"]
        u_obs = joined_uniques.loc[:, "u"]

        t_pdf = t_obs / t_obs.sum()
        u_pdf = u_obs / u_obs.sum()

        return t_pdf, u_pdf
