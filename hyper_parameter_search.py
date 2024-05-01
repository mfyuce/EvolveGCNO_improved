from tqdm import tqdm
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import json 
import time
from graphs.recurrent.graphs_evolvegcn_h_improved import ModelOps
import os
import torch
import torch.multiprocessing as mp
import json
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

MAX_PROCESS = 5 # int(mp.cpu_count()*0.7) #1 #
num_each_threads = int(mp.cpu_count()/MAX_PROCESS)  # 1 #
torch.set_num_threads(num_each_threads)
torch.set_num_interop_threads(num_each_threads)

m_a = -1
m_c = 1
m_p = -1
m_r = -1
m_f = -1
m_m = -1
 

BEST_LAGS  =  [1]#[1,15,16,17,18] ,2,3,4,5
BEST_EDGES  =  [1,2,3,4,5]
BEST_TRAIN_RATIO  =  [0.1,0.2,0.3,0.4]

t = int(time.time())

os.mkdir(f"./runs/{t}")

file_name = f"./runs/{t}/eval_metrics_{t}.csv"
file_name_change = f"./runs/{t}/eval_metrics_change_{t}.csv"


def execute_one(loader, lags, train_ratio, num_train,num_edges, current_try, lr):

    # torch.set_num_threads(num_each_threads)
    # torch.set_num_interop_threads(num_each_threads)

    dataset = loader.get_dataset(lags=lags)
    # device = torch.device('cuda')
    # dataset = dataset.to(device)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
    ml_structure = ModelOps(lr=lr)
    ml_structure.train(loader, train_dataset, num_train=num_train, plot_model=False,
                       calc_perf=True)
    metrics = ml_structure.eval(test_dataset, plot_model=False)
    metrics["nt"] = num_train
    metrics["ne"] = num_edges
    metrics["tr"] = train_ratio
    metrics["l"] = lags
    metrics["lr"] = lr
    # metrics["model"] = ml_structure 
    # if metrics['p']>0.99 or metrics['r']>0.99 or metrics['f']>0.99  or  metrics['a']>0.99  or   metrics['m']>0.99  :
    torch.save(ml_structure.model, f"./runs/{t}/saved_model_{current_try}_{lr}_{ num_edges}_{lags}_{train_ratio}_{num_train }")
    # ml_structure.history1.progress()
    ml_structure.history1.save(f"./runs/{t}/saved_log_{current_try}_{lr}_{ num_edges}_{lags}_{train_ratio}_{num_train }.pkl")
    # ml_structure.plot(["a","p","r","f"])
    # ml_structure.save_after_plot(f"split1/saved_plot_4_{current_try}_{ num_edges}_{lags}_{train_ratio}_{num_train }.png")
    # ml_structure.plot(["a","p","r","f","m"])
    # ml_structure.save_after_plot(f"split1/saved_plot_5_{current_try}_{ num_edges}_{lags}_{train_ratio}_{num_train }.png")

    metrics["current_try"] = current_try
    # q.put(metrics)
    return metrics




with mp.Pool(processes=MAX_PROCESS) as pool:

    with open(file_name, "a+") as outfile:
        with open(file_name_change, "a+") as outfile_change:
            outfile.write("e,l,t,nt,lr,p,r,f,a,c,m\n")
            outfile_change.write("e,l,t,nt,lr,p,r,f,a,c,m\n")
            # for num_edges in (pbar := tqdm( BEST_EDGES)):
            async_results = []
            for num_edges in  [ 5, 1,0,2, 3, 4,50, 100, 150, 200, 10000]: # BEST_EDGES
                loader = BurstAdmaDatasetLoader(num_edges=num_edges,negative_edge=False,features_as_self_edge=True)  # , negative_edge=True)
                for lags in BEST_LAGS:#1, 21
                    # for lags in BEST_LAGS:
                    #for train_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:#
                        train_ratio = 0.7
                        # for train_ratio in BEST_TRAIN_RATIO:
                        for lr in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, \
                                    0.071,0.072,0.073,0.074,0.075,0.076,0.077,0.078,0.079, \
                                    0.081,0.082,0.083,0.084,0.085,0.086,0.087,0.088,0.089, \
                                    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, \
                                    0.1, 0.150, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  #, \   1, 2, 3, 4, 5, 6, 7 , 8, 9
                                
                            for num_train in range(1,10):#
                                # num_train = 20
                                #metrics = execute_one(loader, lags, train_ratio, num_train)
                                #   num_train=1
                                current_loader = loader
                                current_lags = lags
                                current_train_ratio = train_ratio
                                current_num_train = num_train
                                current_num_edges = num_edges
                                current_lr = lr
                                #       execute_one(loader, lags, train_ratio, num_train,num_edges, current_try, lr)
                                async_results.append(
                                    pool.apply_async(
                                        execute_one, args=(
                                            current_loader, current_lags, current_train_ratio, current_num_train,num_edges,t,current_lr)))
                        # pool.join()
            for i, async_result in enumerate(async_results):
                metrics = async_result.get()
                any_change = False
                a = metrics["a"]
                if a > m_a :
                    if a < 1:
                        m_a = a
                    any_change = True
                c = metrics["c"]
                if c < m_c :
                    m_c = c
                    any_change = True
                p = metrics["p"]
                if p > m_p :
                    if p < 1:
                        m_p = p
                    any_change = True
                r = metrics["r"]
                if r > m_r :
                    if r < 1:
                        m_r = r
                    any_change = True
                f = metrics["f"]
                if f > m_f  :
                    if f < 1:
                        m_f = f
                    any_change = True
                m = metrics["m"]
                if m > m_m:
                    m_m = m
                    any_change = True
                if any_change:
                    outfile_change.write(f"{metrics['ne']},{metrics['l']},{metrics['tr']},{metrics['nt']},{metrics['lr']},{metrics['p']},{metrics['r']},{metrics['f']},{metrics['a']},{metrics['c']},{metrics['m']}\n")
                    print("Max So Far")
                    print("m_a\tm_c\tm_p\tm_r\tm_f\tm_m")
                    print(f"{round(m_a, 4)}\t{round(m_c, 4)}\t{round(m_p, 4)}\t{round(m_r, 4)}\t{round(m_f, 4)}\t{round(m_m, 4)}")
                    # torch.save(metrics["model"].model, f"split1/saved_model_{metrics['current_try']}_{metrics['ne']}_{metrics['l']}_{metrics['tr']}_{metrics['nt']}")
                outfile.write(f"{metrics['ne']},{metrics['l']},{metrics['tr']},{metrics['nt']},{metrics['lr']},{metrics['p']},{metrics['r']},{metrics['f']},{metrics['a']},{metrics['c']},{metrics['m']}\n")
                outfile.flush()
                outfile_change.flush()
