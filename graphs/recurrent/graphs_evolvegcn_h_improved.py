import torch
import torch.nn.functional as F
from .evolvegcnh_improved import EvolveGCNHImproved
from sklearn.metrics import precision_recall_fscore_support as score
from . import graphs_base as gb
from torch_geometric.nn import GCNConv

class RecurrentGCN1(torch.nn.Module):
    def __init__(self, node_count, node_features, scriptable=False):
        super(RecurrentGCN1, self).__init__()
        self.scriptable = scriptable
        self.recurrent = EvolveGCNHImproved(node_count, node_features, scriptable=scriptable)
        # if scriptable:
        #     self.recurrent.jittable()

        self.conv1 = GCNConv(node_features, 4)
        if scriptable:
            self.conv1 = self.conv1.jittable()

        self.conv2 = GCNConv(4, 4)
        if scriptable:
            self.conv2 = self.conv2.jittable()

        self.conv3 = GCNConv(4, 2)
        if scriptable:
            self.conv3 = self.conv3.jittable()

        # 2-class logits (benign / malicious) — replaces the old scalar regression head.
        self.linear = torch.nn.Linear(2, 2)
        # if scriptable:
        #     self.linear = self.linear.jittable()

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)

        h = self.conv1(h, edge_index, edge_weight)
        h = h.tanh()
        h = self.conv2(h, edge_index, edge_weight)
        h = h.tanh()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv3(h, edge_index, edge_weight)
        out = h.tanh()  # Final GNN embedding space.

        h = self.linear(out)   # (N, 2) — classification logits
        return h, out

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_count, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNHImproved(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        # h = F.relu(h)
        # h = F.relu6(h)
        h = F.selu(h)
        h = self.linear(h)
        return h,None
 


class ModelOps(gb.BaseGrafModelOps):
    def __init__(self, lr=0.01) -> None:
        super().__init__(lr=lr)
    
    def snapshot_train(self): 
        y_hat, out =  self.model(self.snapshot.x, self.snapshot.edge_index, self.snapshot.edge_attr )
        return y_hat
    
    def train(self, loader,train_dataset, num_train=1,plot_model=False, calc_perf=True, scriptable=False):
        number_of_nodes=len(loader._dataset["node_labels"])
        model = RecurrentGCN1(node_features=loader.n_node_features, node_count=number_of_nodes, scriptable=scriptable)
        
        if scriptable:
            model = torch.jit.script(model)

        return super().train(loader,
                             train_dataset, 
                             model,
                             num_train=num_train,
                             plot_model=plot_model, 
                             calc_perf=calc_perf)
      
