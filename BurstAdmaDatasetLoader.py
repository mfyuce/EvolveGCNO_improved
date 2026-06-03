import io
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.signal  import DynamicGraphTemporalSignal
import os

class BurstAdmaDatasetLoader(object):
    """A dataset of mobility and history of reported cases of COVID-19
    in England NUTS3 regions, from 3 March to 12 of May. The dataset is
    segmented in days and the graph is directed and weighted. The graph
    indicates how many people moved from one region to the other each day,
    based on Facebook Data For Good disease prevention maps.
    The node features correspond to the number of COVID-19 cases
    in the region in the past **window** days. The task is to predict the
    number of cases in each node after 1 day. For details see this paper:
    `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_
    """

    def __init__(self, num_edges=0, negative_edge=False, features_as_self_edge=False, dataset=None, binary=True):
        self.num_edges = num_edges
        self.negative_edge = negative_edge
        self.features_as_self_edge = features_as_self_edge
        # binary=True  -> 0 benign / 1 any-attacker (2-class node classification)
        # binary=False -> raw 8-class attacker type (0..7) kept as-is
        self.binary = binary
        if dataset is None:
            self._read_web_data()
        else :
            self._dataset = dataset
    def _read_web_data(self):
        file_name = ""
        self_edges = '_with_features_as_self_edge' if self.features_as_self_edge else ''
        if self.num_edges >= 0:
            file_name = f"{os.path.dirname(os.path.realpath(__file__))}/data/myoutput_{self.num_edges}_edges{'_negative' if self.negative_edge else '_positive'}{self_edges}.json"
        else:
            file_name = f"{os.path.dirname(os.path.realpath(__file__))}/data/myoutput.json"
        # print(os.path.dirname(os.path.realpath(__file__)))
        with open(file_name, "r") as outfile:
            self._dataset = json.load(outfile)

    def _get_edges(self):
        self._edges = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edges.append(
                np.array(self._dataset["edge_index"][str(time)]).T
            )

    def _get_edge_weights(self):
        self._edge_weights = []
        for time in range(self._dataset["time_periods"] - self.lags):
            self._edge_weights.append(
                np.array(self._dataset["edge_weight"][str(time)])
            )

    def _get_targets_and_features(self):

        stacked_target   = np.array(self._dataset["y"])        # (T, N)
        stacked_features = np.array(self._dataset["features"]) # (T, N, F) or (T, N)

        # Standardize features along the time axis; works for both 2-D and 3-D arrays.
        f_mean = np.mean(stacked_features, axis=0, keepdims=True)
        f_std  = np.std(stacked_features,  axis=0, keepdims=True)
        standardized_features = (stacked_features - f_mean) / (f_std + 1e-10)

        if stacked_features.ndim == 3:
            # Multi-feature case: (T, N, F) — one (N, F) slice per snapshot.
            # Temporal context is handled by the EvolveGCN-H GRU, so lags are not
            # needed here; the current-step features are used directly.
            self.features = [
                standardized_features[i, :, :]          # (N, F)
                for i in range(self._dataset["time_periods"] - self.lags)
            ]
        else:
            # Scalar feature case (COVID-style): slide a window of length `lags`.
            self.features = [
                standardized_features[i : i + self.lags, :].T   # (N, lags)
                for i in range(self._dataset["time_periods"] - self.lags)
            ]

        # Classification targets (int64 so CrossEntropyLoss consumes them directly).
        #   binary=True  -> 0 benign / 1 any-attacker
        #   binary=False -> raw 8-class attacker type (0..7)
        if self.binary:
            label_array = (stacked_target != 0).astype(np.int64)
        else:
            label_array = stacked_target.astype(np.int64)
        self.targets = [
            label_array[i + self.lags, :]               # (N,) int64
            for i in range(self._dataset["time_periods"] - self.lags)
        ]

    @property
    def n_node_features(self) -> int:
        """Number of kinematic features per node (F in the (T, N, F) array)."""
        f = np.array(self._dataset["features"])
        return int(f.shape[-1]) if f.ndim == 3 else 1

    def get_dataset(self, lags: int = 8) -> DynamicGraphTemporalSignal:
        """Returning the England COVID19 data iterator.

        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The England Covid dataset.
        """
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        dataset = DynamicGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
