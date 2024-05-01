# Install required packages.
import os
import torch
# os.environ['TORCH'] = torch.__version__
# print(torch.__version__)

# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
# !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# Helper function for visualization.
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()



import os 
import torch


import graphs.recurrent.graphs_base as base
# from graphs.recurrent.graphs_agcrn import  ModelOps
# from graphs.recurrent.graphs_dcrnn import ModelOps
# from graphs.recurrent.graphs_a3tcgn import ModelOps
# from graphs.recurrent.graphs_dygrencoder import ModelOps
# from graphs.recurrent.graphs_evolvegcn_h import ModelOps
from graphs.recurrent.graphs_evolvegcn_h_improved import ModelOps
# from graphs.recurrent.graphs_evolvegcn_o import ModelOps
# from graphs.recurrent.graphs_gclstm import ModelOps
# from graphs.recurrent.graphs_gconvgru import ModelOps
# from graphs.recurrent.graphs_lrgcn import ModelOps
# from graphs.recurrent.graphs_mpnnlstm import ModelOps
# from graphs.recurrent.graphs_tgcn import ModelOps
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import multiprocessing as mp
num_each_threads = 1
torch.set_num_threads(num_each_threads)
torch.set_num_interop_threads(num_each_threads)

from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
# from autonotebook import tqdm as notebook_tqdm

loader = BurstAdmaDatasetLoader(num_edges=5, negative_edge=False, features_as_self_edge=True)

dataset = loader.get_dataset(lags=1)

from torch_geometric_temporal.signal import temporal_signal_split

# device = torch.device('cuda')
# dataset = dataset.to(device)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
train_dataset._target_mean = loader._target_mean
train_dataset._target_std = loader._target_std

test_dataset._target_mean = loader._target_mean
test_dataset._target_std = loader._target_std


# train_snapshots = int(0.7 * dataset.snapshot_count)

# test_dataset._original_target = loader._original_target
# test_dataset.test_starts_at = train_snapshots+1


# train_dataset._original_target = loader._original_target
# train_dataset.train_ends_at = train_snapshots

import os 
import torch
#https://stackoverflow.com/a/63922083/1290868
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# best model so far
ml_structure =ModelOps() 
#train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
# ml_structure.model = torch.load("../split1/l_1-20_nt_1-19/models/saved_model_1681132533_0_20_0.7_20")
# ml_structure.model = torch.load("../split1/saved_model_1681142690_3_20_0.7_2")
## best model so far
# ml_structure.model = torch.load("../split1/saved_model_1681219225_1_4_0.7_11")
# ml_structure.model = torch.load("../split1/saved_model_1681219225_5_2_0.7_2")
# ml_structure.model = torch.load("../split1/lr_test_2/saved_model_1681301082_0.8_5_2_0.7_20")



#ml_structure.model = torch.load("model/saved_model_1681336502_0.007_5_1_0.7_4" )

##w
# /home/ulak/egcne/EvolveGCNO_improved/runs/1714566510/saved_log_1714566510_0.7_0_1_0.7_2.pkl
# /home/ulak/egcne/EvolveGCNO_improved/runs/1714566510/saved_model_1714566510_0.7_0_1_0.7_2
ml_structure.model = torch.load("/home/ulak/egcne/EvolveGCNO_improved/runs/1714566510/saved_model_1714566510_0.7_0_1_0.7_2" )



base.SCORE_METHOD = "weighted"
#with torch.autograd.profiler.profile(use_cuda=False) as prof:
#metrics = ml_structure.eval(test_dataset ,plot_model=True)

import warnings
warnings.filterwarnings('ignore') 

all_metrics = []
for score_type in ["micro"]: #"weighted","micro", "macro", ,None (Per class perf), "samples", "binary"
    base.SCORE_METHOD = score_type
    print(f"Scoring with :{score_type}")
    metrics = ml_structure.eval(test_dataset ,plot_model=False)
    all_metrics.append({"score_type":score_type,"metrics":metrics});




print("\n".join(t["score_type"] + str(t["metrics"]) for t in all_metrics))




# ml_structure.plot()
# print(metrics)
# ml_structure.save_model_visuals(f"../split1/torchviz_eval_{ml_structure.time}_saved_model_1681301082_0.8_5_2_0.7_20",\
#                                 "../split1/saved_model_1681301082_0.8_5_2_0.7_20",\
#                                     ml_structure.snapshot_eval())