import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay

import numpy as np

import hiddenlayer as hl
from tqdm import tqdm
# criterion = torch.nn.CrossEntropyLoss()
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import (
    ExplainerConfig,
    ExplanationType,
    ModelConfig,
    ModelMode,
    ThresholdConfig,
    ThresholdType,
)
SCORE_METHOD = "weighted" # None, "micro", "macro", "samples", "weighted", "binary", "binary"

class BaseGrafModelOps():
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        self.acc=None
        self.cost = None
        self.p = 0
        self.r = 0
        self.f = 0
        self.m = 0
        self.s = 0
        self.p_a = []
        self.epoch = 0
        self.snapshot = 0
        self.time = 0
        self.loader = None
        self.model = None
        self.num_train = None
        self.extras={}
        self.dots = None
        self.plot_index=0
        # A History object to store metrics
        self.history1 = hl.History()
        # A Canvas object to draw the metrics
        self.canvas1 = hl.Canvas()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss = torch.nn.CrossEntropyLoss()  # Define loss criterion.

    def snapshot_train(self):
        return self.model(self.snapshot.x, self.snapshot.edge_index, self.snapshot.edge_attr)

    def snapshot_eval(self):
        return self.model(self.snapshot.x, self.snapshot.edge_index, self.snapshot.edge_attr)

    def snapshot_epoch(self, epoch ):
        self.cost = 0
        self.acc = 0
        self.epoch = epoch 
        self.acc = 0
        self.acc1 = 0
        self.p = 0
        self.r = 0
        self.f = 0 
        self.m = 0 
        self.s = 0


        self.step_cost = 0
        self.step_acc = 0
        self.step_epoch = epoch 
        self.step_acc = 0
        self.step_acc1 = 0
        self.step_p = 0
        self.step_r = 0
        self.step_f = 0 
        self.step_m = 0 
        self.step_s = 0
        self.cm = {}

    def eval_one(self,snapshot,time,plot_model):
        self.snapshot = snapshot
        self.time = time  
        model_plotted = False
        y_hat = self.snapshot_eval()
        if type(y_hat) is tuple:
            y_hat = y_hat[0]
        if plot_model and not model_plotted:
            self.save_model_visuals(f"./runs/plot/torchviz_eval_{time}","./runs/plot/saved_model",y_hat)
            # from torchviz import make_dot
            # plot_file_name = f"torchviz_eval_{time}"
            # # self.dots.append(make_dot(y_hat, params=dict(list(model.named_parameters()))).render(f"rnn_torchviz_{time}_{epoch}", format="png", outfile=f"rnn_torchviz_{time}_{epoch}.svg", show_attrs=True, show_saved=True))
            # self.dots = make_dot(y_hat, params=dict(list(self.model.named_parameters()))).render(f"{plot_file_name}.dot", format="png", outfile=f"{plot_file_name}.png")
            model_plotted = True
            # torch.save(self.model, "./saved_model", _use_new_zipfile_serialization=False)
            # import hiddenlayer as hl
            # self.dots = hl.build_graph(self.model, self.snapshot.edge_index, self.snapshot.edge_attr)
        
        pred = y_hat.argmax(dim=1)
        detached_y = (snapshot.y.detach().cpu().numpy()*10.0%8).round()
        detached_y_hat =  (y_hat.detach().cpu().numpy()*10.0%8).round()
        correct = np.count_nonzero((pred.detach().cpu().numpy()*10.0%8).round() == detached_y)
        all = len(snapshot.y)

        self.step_m = matthews_corrcoef(detached_y, detached_y_hat)
        self.step_acc = accuracy_score(detached_y, detached_y_hat) #correct /  all 
        self.step_acc1 = correct/all

        self.m += self.step_m
        self.acc += self.step_acc #correct /  all 
        self.acc1 += self.step_acc1
        # try:
        if not SCORE_METHOD:
            t = score(detached_y,  detached_y_hat, average=SCORE_METHOD,zero_division=1)
            print (t)
        else:
            precision, recall, fscore, support = score(detached_y,  detached_y_hat, average=SCORE_METHOD,zero_division=1)

        self.step_p = precision 
        self.step_r = recall 
        self.step_f = fscore 
        self.step_s = 0 if support is None else support 


        self.p += self.step_p 
        self.r += self.step_r 
        self.f += self.step_f 
        self.s += self.step_s
        # except:
            # num_minus+=1 

        # precision1, recall1, fscore1, support1 = score(snapshot.y.round().detach().numpy(),  \
        #                                            y_hat.round().detach().numpy(), average=SCORE_METHOD,zero_division=1,\
        #                                             labels=self.loader._dataset["node_labels"])
            
        self.step_cost =  torch.mean((y_hat-snapshot.y)**2)
        self.cost += self.step_cost
        # self.cost = self.cost + criterion(y_hat, snapshot.y)
        # # Store metrics in the history object
        # self.plot_index+=1
        # history1.log(self.plot_index, c=self.cost/ (time+1), \
        #                 a=self.acc / (time+1), \
        #                 p=self.p / (time+1), \
        #                 r=self.r / (time+1), \
        #                 f=self.f / (time+1), \
        #                 m=self.m / (time+1), \
        #                 s=self.s)

    def train(self, loader,train_dataset, model,num_train=1, plot_model=True, calc_perf=True):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr) 

        model.train() 
        self.loader = loader
        self.model = model
        self.num_train = num_train

        # model_plotted = False
        num_minus = 0
        # for epoch in (pbar := tqdm(range(num_train))):
        for epoch in range(num_train):
        # for epoch in range(num_train):

            self.snapshot_epoch(epoch)
            
            for time, snapshot in enumerate(train_dataset):
                self.eval_one(snapshot,time,plot_model)
                # self.time = time
                # self.snapshot = snapshot

                # y_hat = self.snapshot_train()

                # if type(y_hat) is tuple:
                #     y_hat = y_hat[0]
                # # if plot_model and not model_plotted:
                # #     from torchviz import make_dot
                # #     plot_file_name = f"torchviz_train_{time}_{epoch}"
                # #     #self.dots.append(make_dot(y_hat, params=dict(list(model.named_parameters()))).render(f"rnn_torchviz_{time}_{epoch}", format="png", outfile=f"rnn_torchviz_{time}_{epoch}.svg", show_attrs=True, show_saved=True))
                # #     make_dot(y_hat, params=dict(list(model.named_parameters()))).render(f"{plot_file_name}.dot", format="png", outfile=f"{plot_file_name}.png")
                # #     model_plotted = True

                # diff_tensor = y_hat-snapshot.y
                # # diff = torch.sum(torch.abs(diff_tensor))
                # self.cost = self.cost + torch.mean(diff_tensor**2)
                # # self.cost = self.cost + criterion(y_hat, snapshot.y)
                # # self.cost = self.loss(y_hat.round(), snapshot.y.round())
                # if calc_perf:
                #     pred = y_hat.argmax(dim=1)
                #     correct = (pred.detach().cpu().numpy().round() == snapshot.y.detach().cpu().numpy()).sum()

                #     all = len(snapshot.y)
                #     # print(f'correct: {correct:.4f} all: {all:.4f}')
                #     detached_y = snapshot.y.detach().cpu().numpy().round()
                #     detached_y_hat =  y_hat.detach().cpu().numpy().round()
                #     self.acc += accuracy_score(detached_y, detached_y_hat) #correct / int(all)
                #     self.m +=  matthews_corrcoef(detached_y, detached_y_hat)

                #     self.acc1 += correct / all
                #     try:
                #         precision, recall, fscore, support = score(detached_y,  detached_y_hat, average=SCORE_METHOD,zero_division=1)

                #         self.p += precision 
                #         self.r += recall 
                #         self.f += fscore  
                #         self.s += 0 if support is None else support  
                #     except:
                #         num_minus+=1
                # p_1 = self.p / (time+1)
                # r_1 = self.r / (time+1)
                # f_1 = self.f / (time+1)
                # pbar.set_description(f'accuracy : {acc/ (time+1)} MSE: {cost / (time+1)} precision: {p_1} recall: {r_1} f1: {f_1} support:{support}')
                
                # precision, recall, fscore, support = score(snapshot.y.round().detach().numpy(),  y_hat.round().detach().numpy(), average=None)
                # p_a.append(precision)
                # cur_cost = cost / (time+1)
                # pbar.set_description(f'cost: {round(diff.item(),2)} MSE: {round(cur_cost.item(),2)}')
                # Store metrics in the history object
                # self.plot_index+=1
                # history1.log(self.plot_index, c=self.cost/ (time+1), \
                #                 a=self.acc / (time+1), \
                #                 p=self.p / (time+1), \
                #                 r=self.r / (time+1), \
                #                 f=self.f / (time+1), \
                #                 m=self.m / (time+1), \
                #                 s=self.s)
                

            if calc_perf:
                result_after_min = time-num_minus+1
                result_after_min = (1 if result_after_min == 0 else (time-num_minus+1))
                self.p = self.p / result_after_min
                self.r = self.r / result_after_min
                self.f = self.f / result_after_min
                self.s = self.s / result_after_min
            
                self.m = self.m  / (time+1)
                self.acc = self.acc / (time+1)
                self.acc1 = self.acc1 / (time+1)
                if plot_model:
                    self.plot_index+=1
                    self.history1.log(self.plot_index, \
                                    c=self.cost / (time+1), \
                                    a=self.acc , \
                                    a1=self.acc1 , \
                                    p=self.p , \
                                    r=self.r , \
                                    f=self.f , \
                                    m=self.m , \
                                    s=self.s)
            self.cost = self.cost / (time+1)
            # if plot_model:
            #     self.plot(["a","p","r","f"])
            self.cost.backward()
            optimizer.step()
            optimizer.zero_grad()
            if calc_perf:
                self.p_a.append({'p':self.p,'r':self.r,'f':self.f,"a":self.acc,"m":self.m, "c":self.cost})
            # pbar.set_description(f"cost: {diff.item()} MSE: {cost}")
            # pbar.set_description(f"MSE: {self.cost}")
            # pbar.set_description(f"accuracy : {acc} MSE: {cost} precision: {p} recall: {r} f1: {f} support:{support}")
        return self

    def _get_real_targets(self,  dataset, targets):
        return targets * (dataset._target_std+ 10 ** -10)  +  dataset._target_mean
    def sum_np(self, a):
        t=0
        for x in a:
            t+=x
        return t
    def eval(self, test_dataset, plot_model=True, explain=True):
        self.model.eval()

        self.snapshot_epoch(0)
        model_plotted = False
        num_minus = 0
        self.cm1 = [[0 for x in range(8)] for y in range(8)]
        for time, snapshot in enumerate(test_dataset):
            
            self.snapshot = snapshot
            self.time = time  
            y_hat = self.snapshot_eval()
            if type(y_hat) is tuple:
                y_hat = y_hat[0]
            if plot_model and not model_plotted:
                self.save_model_visuals(f"./runs/plot/torchviz_eval_{time}","./runs/plot/saved_model",y_hat)
                # from torchviz import make_dot
                # plot_file_name = f"torchviz_eval_{time}"
                # # self.dots.append(make_dot(y_hat, params=dict(list(model.named_parameters()))).render(f"rnn_torchviz_{time}_{epoch}", format="png", outfile=f"rnn_torchviz_{time}_{epoch}.svg", show_attrs=True, show_saved=True))
                # self.dots = make_dot(y_hat, params=dict(list(self.model.named_parameters()))).render(f"{plot_file_name}.dot", format="png", outfile=f"{plot_file_name}.png")
                model_plotted = True
                # torch.save(self.model, "./saved_model", _use_new_zipfile_serialization=False)
                # import hiddenlayer as hl
                # self.dots = hl.build_graph(self.model, self.snapshot.edge_index, self.snapshot.edge_attr)
            
            pred = y_hat.argmax(dim=1)
            detached_y = (snapshot.y.detach().cpu().numpy()*10.0%8).round()
            detached_y_hat =  (y_hat.detach().cpu().numpy()*10.0%8).round()
            correct = np.count_nonzero((pred.detach().cpu().numpy()*10.0%8).round() == detached_y)
            all = len(snapshot.y)
            self.m += matthews_corrcoef(detached_y, detached_y_hat)
            self.acc += accuracy_score(detached_y, detached_y_hat) #correct /  all 
            self.acc1 += correct/all
            #print(f"{time}:{classification_report(detached_y, detached_y_hat)}")#,labels=["0","1","2","3","4","5","6","7"]
            # cm = confusion_matrix(detached_y, detached_y_hat).tolist()
            # l = len(cm)
            # import random 
            # for i in range(l):
            #     for j in range(l):
            #         k = random.randrange(8) 
            #         self.cm1[(i+k)%8][(j+k)%8] +=cm[i][j]
            # from pycm import ConfusionMatrix    

            # cm1 = ConfusionMatrix(actual_vector=detached_y,predict_vector=detached_y_hat)
            # print(cm1)

            # _FP = self.sum_np((cm.sum(axis=0) - np.diag(cm)).tolist()) 
            # _FN = self.sum_np((cm.sum(axis=1) - np.diag(cm)).tolist()) 
            # _TP = self.sum_np(np.diag(cm).tolist() ) 
            # _ALL = self.sum_np(np.concatenate(cm).tolist())
            # if self.cm.get("FP") is None:
            #     self.cm["FP"] = _FP
            # else:
            #     self.cm["FP"] += _FP
            
            # if self.cm.get("FN")is None:
            #     self.cm["FN"] = _FN
            # else:
            #     self.cm["FN"] += _FN
            
            # if self.cm.get("TP")is None:
            #     self.cm["TP"] = _TP
            # else:
            #     self.cm["TP"] += _TP
            # _TN = _ALL - (_FP + _FN + _TP)
            # if _TN<0:
            #     pass
            # if self.cm.get("TN")is None:
            #     self.cm["TN"] = _TN
            # else:
            #     self.cm["TN"] += _TN

            # o = {}
            # o["time"] = time
            # o["cm"] = cm.tolist()
            # import json
            # print(f"{json.dumps(o)}")#,labels=["0","1","2","3","4","5","6","7"]
            # import matplotlib.pyplot as plt
            # cm_display = ConfusionMatrixDisplay(confusion_matrix = cm)#, display_labels = [0, 1]

            # cm_display.plot()
            # plt.show()
            # print(f"{time}:{classification_report(self._get_real_targets(test_dataset,detached_y), self._get_real_targets(test_dataset,detached_y_hat))}")#,labels=["0","1","2","3","4","5","6","7"]
            # try:
            if not SCORE_METHOD:
                precision, recall, fscore, support = score(detached_y,  detached_y_hat, average=SCORE_METHOD,zero_division=1)
                print (precision, recall, fscore, support)
            else:
                precision, recall, fscore, support = score(detached_y,  detached_y_hat, average=SCORE_METHOD,zero_division=1)
            
            # original_y = test_dataset._original_target[test_dataset.test_starts_at + time ]
            self.p += precision 
            self.r += recall 
            self.f += fscore 
            self.s += 0 if support is None else support 
            # except:
                # num_minus+=1 

            # precision1, recall1, fscore1, support1 = score(snapshot.y.round().detach().numpy(),  \
            #                                            y_hat.round().detach().numpy(), average=SCORE_METHOD,zero_division=1,\
            #                                             labels=self.loader._dataset["node_labels"])
                
            self.cost = self.cost + torch.mean((y_hat-snapshot.y)**2)
            # self.cost = self.cost + criterion(y_hat, snapshot.y)
            # # Store metrics in the history object
            # self.plot_index+=1
            # history1.log(self.plot_index, c=self.cost/ (time+1), \
            #                 a=self.acc / (time+1), \
            #                 p=self.p / (time+1), \
            #                 r=self.r / (time+1), \
            #                 f=self.f / (time+1), \
            #                 m=self.m / (time+1), \
            #                 s=self.s)

        result_after_min = time-num_minus+1
        result_after_min = (1 if result_after_min == 0 else (time-num_minus+1))
        self.p = self.p / result_after_min
        self.r = self.r / result_after_min
        self.f = self.f / result_after_min
        self.s = self.s / result_after_min

        self.acc = self.acc / (time+1)
        self.m = self.m / (time+1)
        self.acc1 = self.acc1 / (time+1)
        self.cost = self.cost / (time+1)
        
        # # https://stackoverflow.com/a/50671617/1290868
        # # self.cm["FP"] = self.cm["FP"].astype(float)
        # # self.cm["FN"] = self.cm["FN"].astype(float)
        # # self.cm["TP"] = self.cm["TP"].astype(float)
        # # self.cm["TN"] = self.cm["TN"].astype(float)
        # cm2 = np.array(self.cm1)
        # _FP = self.sum_np((cm2.sum(axis=0) - np.diag(cm2)).tolist()) 
        # _FN = self.sum_np((cm2.sum(axis=1) - np.diag(cm2)).tolist()) 
        # _TP = self.sum_np(np.diag(cm2).tolist() ) 
        # _ALL = self.sum_np(np.concatenate(cm2).tolist())
        # if self.cm.get("FP") is None:
        #     self.cm["FP"] = _FP
        # else:
        #     self.cm["FP"] += _FP

        # if self.cm.get("FN")is None:
        #     self.cm["FN"] = _FN
        # else:
        #     self.cm["FN"] += _FN

        # if self.cm.get("TP")is None:
        #     self.cm["TP"] = _TP
        # else:
        #     self.cm["TP"] += _TP
        # _TN = _ALL - (_FP + _FN + _TP)
        # if _TN<0:
        #     pass
        # if self.cm.get("TN")is None:
        #     self.cm["TN"] = _TN
        # else:
        #     self.cm["TN"] += _TN
        # # Sensitivity, hit rate, recall, or true positive rate
        # self.cm["TPR"] = self.cm["TP"]/(self.cm["TP"]+self.cm["FN"])
        # # Specificity or true negative rate
        # self.cm["TNR"] = self.cm["TN"]/(self.cm["TN"]+self.cm["FP"]) 
        # # Precision or positive predictive value
        # self.cm["PPV"] = self.cm["TP"]/(self.cm["TP"]+self.cm["FP"])
        # # Negative predictive value
        # self.cm["NPV"] = self.cm["TN"]/(self.cm["TN"]+self.cm["FN"])
        # # Fall out or false positive rate
        # self.cm["FPR"] = self.cm["FP"]/(self.cm["FP"]+self.cm["TN"])
        # # False negative rate
        # self.cm["FNR"] = self.cm["FN"]/(self.cm["TP"]+self.cm["FN"])
        # # False discovery rate
        # self.cm["FDR"] = self.cm["FP"]/(self.cm["TP"]+self.cm["FP"])
        # # Overall accuracy
        # self.cm["OverallACC"] = (self.cm["TP"]+self.cm["TN"])/(self.cm["TP"]+self.cm["FP"]+self.cm["FN"]+self.cm["TN"])
        
        if plot_model:
            self.plot_index+=1
            self.history1.log(self.plot_index, c=self.cost, \
                            a=self.acc , \
                            a1=self.acc1 , \
                            p=self.p , \
                            r=self.r , \
                            f=self.f , \
                            m=self.m , \
                            s=self.s)
        # self.plot()
        # if explain:
        #     explainer = Explainer(
        #         model=self.model,
        #         algorithm=GNNExplainer(epochs=200),
        #         explainer_config=ExplainerConfig(explanation_type='model',
        #                                             node_mask_type='attributes',
        #                                             edge_mask_type='object'),
        #         model_config=ModelConfig(
        #             mode='regression', #'multiclass_classification',
        #             task_level='node',
        #             return_type='log_probs',  # Model returns log probabilities.
        #         ),
        #     )

        #     # Generate explanation for the node at index `10`:
        #     explanation = explainer(snapshot.x, snapshot.edge_index,  index=10,edge_weight=snapshot.edge_weight)
        #     print(f"Explanation edge_mask: {explanation.edge_mask}")
        #     print(f"Explanation node_mask: {explanation.node_mask}")
        #     explanation.visualize_feature_importance(top_k=10)

        #     explanation.visualize_graph()
        #     try:
        #         from  torch_geometric.explain import unfaithfulness # tpyg 2.3
        #         metric = unfaithfulness(explainer, explanation)
        #         print(f"Explanation unfaithfulness: {metric}")
        #     except:
        #         pass
        return {"p":self.p,"r":self.r,"f":self.f,"a":self.acc,"a1":self.acc1,"m":self.m, "c":self.cost.item(), "cm":self.cm}
    def plot(self, fields=[]):
        # Plot the two metrics in one graph
        arr=[]
        if fields:
            for field in fields:
                arr.append(self.history1[field])
        else:
            arr = [self.history1["a"], self.history1["a1"], self.history1["p"], self.history1["r"], self.history1["f"], \
                                self.history1["s"], self.history1["c"], self.history1["m"]]
        self.canvas1.draw_plot(arr)

    def save_after_plot(self, save_path):
        self.canvas1.save(save_path)

    def save_model_visuals(self, plot_file_name,plot_model_file_name, y_hat):
        from torchviz import make_dot
        # self.dots.append(make_dot(y_hat, params=dict(list(model.named_parameters()))).render(f"rnn_torchviz_{time}_{epoch}", format="png", outfile=f"rnn_torchviz_{time}_{epoch}.svg", show_attrs=True, show_saved=True))
        self.dots = make_dot(y_hat, params=dict(list(self.model.named_parameters()))).render(f"{plot_file_name}.dot", format="png", outfile=f"{plot_file_name}.png")
        torch.save(self.model, plot_model_file_name, _use_new_zipfile_serialization=False)

