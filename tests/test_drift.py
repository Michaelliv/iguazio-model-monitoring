import yaml
from monitoring import drift


class TestVirtualDrift:

    # Get histogram file
    with open("tests/model_hist_v1.yaml", "r") as f:
        base_histogram = yaml.safe_load(f)
    with open("tests/model_hist_v2.yaml", "r") as f:
        latest_histogram = yaml.safe_load(f)

    def test_vd_same_histogram(self):
        # Extract feature histograms
        vd = drift.VirtualDrift(inf_capping=10)

        drift_results = vd.compute_drift_from_histograms(
            self.base_histogram, self.base_histogram
        )
        print(drift_results)
        assert sum(list(drift_results.values())) == 0

    def test_vd_different_histogram(self):
        # Extract feature histograms
        vd = drift.VirtualDrift(inf_capping=10)

        drift_results = vd.compute_drift_from_histograms(
            self.base_histogram, self.latest_histogram
        )
        print(drift_results)
        assert sum(list(drift_results.values())) != 0
