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

    def __init__(self, num_edges=0, negative_edge=False, features_as_self_edge=False, dataset=None):
        self.num_edges = num_edges
        self.negative_edge = negative_edge
        self.features_as_self_edge = features_as_self_edge
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

        stacked_target = np.array(self._dataset["y"])
        self._target_std= np.std(stacked_target, axis=0)
        self._target_mean= np.mean(stacked_target, axis=0)
        self._original_target= np.array(self._dataset["y"])


        stacked_features = np.array(self._dataset["features"])
        
        #TODO: stacked_target!
        
        # standardized_features = (stacked_features - np.mean(stacked_features, axis=0)) / (
        #     np.std(stacked_features, axis=0) + 10 ** -10
        # )
        # standardized_target = (stacked_target - self._target_mean) / (
        #     self._target_std + 10 ** -10
        # )
        standardized_target = stacked_target / 10.0
        # self.features = [
        #     standardized_features[i : i + self.lags, :].T
        #     for i in range(self._dataset["time_periods"] - self.lags)
        # ]

        self.features = [
            standardized_target[i : i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
        self.targets = [
            standardized_target[i + self.lags, :].T
            for i in range(self._dataset["time_periods"] - self.lags)
        ]
    
    def _get_real_targets(self,  dataset, targets):
        return targets * (dataset._target_std+ 10 ** -10)  +  dataset._target_mean
        
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
