import re
from typing import List

import pandas as pd
from sklearn import datasets

from src.drift import VirtualDrift

ALPHA_NUM_RE = re.compile("[^0-9a-zA-Z ]+")


def get_iris_feature_cols() -> List[str]:
    iris = datasets.load_iris()
    return list(map(lambda x: ALPHA_NUM_RE.sub("", x).replace(" ", "_"), iris.feature_names))


def get_iris_df() -> pd.DataFrame:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=get_iris_feature_cols())
    df["label"] = iris.target
    return df


class TestVirtualDrift:

    def test_compute_drift(self):
        df = get_iris_df()
        vd = VirtualDrift(5, get_iris_feature_cols(), None, "label")
        pass

    def test_init_discritizers(self):
        assert False

    def test_fit_discretizer(self):
        assert False

    def test_discritize(self):
        assert False

    def test_compute_drift_measures(self):
        assert False

    def test_compute_feature_drift(self):
        assert False

    def test_compute_prediction_drift(self):
        assert False

    def test_compute_label_drift(self):
        assert False

    def test_to_observations(self):
        base_data = get_iris_df().sample(frac=1).reset_index(drop=True)
        base_data = pd.DataFrame(base_data.loc[:, "label"])

        current_data = get_iris_df().sample(frac=1).reset_index(drop=True).loc[:"label"]
        current_data = pd.DataFrame(current_data.loc[:, "label"])

        base_labels, current_labels = VirtualDrift.to_observations(base_data, current_data)

        assert base_labels["0"].round(1) == 0.3
        assert current_labels["0"].round(1) == 0.3
