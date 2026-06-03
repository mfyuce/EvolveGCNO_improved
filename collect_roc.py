"""Train key models on the vehicle-disjoint split (seed 3) and save test
(probability, label) arrays for ROC curves.  Run one model per call.

Run:  python collect_roc.py <gconvgru|static|evolveo|rf>
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
import graphs.recurrent.graphs_base as base

M = sys.argv[1] if len(sys.argv) > 1 else "gconvgru"
SEED, EP, WIN, H, LR = 3, 40, 24, 32, 0.01

class GG(nn.Module):
    def __init__(s, f): super().__init__(); s.r=GConvGRU(f,H,2); s.l=nn.Linear(H,H); s.c=nn.Linear(H,2)
    def forward(s,x,ei,ew,h=None):
        h=s.r(x,ei,ew,h); return s.c(F.dropout(F.relu(s.l(h)),0.5,s.training)), h
class SG(nn.Module):
    def __init__(s, f): super().__init__(); s.a=GCNConv(f,H); s.b=GCNConv(H,H); s.d=GCNConv(H,H); s.c=nn.Linear(H,2)
    def forward(s,x,ei,ew):
        h=F.relu(s.a(x,ei,ew)); h=F.relu(s.b(h,ei,ew)); h=F.dropout(h,0.5,s.training); return s.c(F.relu(s.d(h,ei,ew)))
class EO(nn.Module):
    def __init__(s, f): super().__init__(); s.r=EvolveGCNO(f); s.a=GCNConv(f,H); s.b=GCNConv(H,H); s.d=GCNConv(H,H); s.c=nn.Linear(H,2)
    def reset(s):
        try: s.r.weight=None
        except Exception: object.__setattr__(s.r,"weight",None)
    def forward(s,x,ei,ew):
        h=s.r(x,ei,ew); h=F.relu(s.a(h,ei,ew)); h=F.relu(s.b(h,ei,ew)); h=F.dropout(h,0.5,s.training); return s.c(F.relu(s.d(h,ei,ew)))

def focal(lg,t,a,g=2.0):
    lp=F.log_softmax(lg,-1).gather(1,t.unsqueeze(1)).squeeze(1)
    return (((1-lp.exp())**g)*(-lp)*a.to(lg.device)[t]).mean()

lb=BurstAdmaDatasetLoader(num_edges=5,negative_edge=False,features_as_self_edge=True)
ds=lb.get_dataset(lags=1); alls=list(ds)
T,lags,N=lb._dataset["time_periods"],lb.lags,len(lb._dataset["node_labels"])
aug=np.load("data/features_augmented.npy"); active=aug[...,0]!=0.0
flat=aug.reshape(-1,10); augN=((aug-flat.mean(0))/(flat.std(0)+1e-8))
Y=np.stack([lb.targets[i] for i in range(T-lags)]); node_label=(Y.max(0)>0).astype(int)
rng=np.random.default_rng(SEED); trm=np.zeros(N,bool)
for c in (0,1):
    idx=np.where(node_label==c)[0]; rng.shuffle(idx); trm[idx[:int(0.7*len(idx))]]=True
trv,tev=torch.tensor(trm),torch.tensor(~trm); n_tr=int(0.7*(T-lags))

if M=="rf":
    Xtr,ytr,Xte,yte=[],[],[],[]
    for i in range(T-lags):
        a=active[i]
        if i<n_tr: m=a&trm; Xtr.append(augN[i][m]); ytr.append(Y[i][m])
        m2=a&(~trm); Xte.append(augN[i][m2]); yte.append(Y[i][m2])
    rf=RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=4,random_state=SEED)
    rf.fit(np.concatenate(Xtr),np.concatenate(ytr))
    p=rf.predict_proba(np.concatenate(Xte))[:,1]; l=np.concatenate(yte)
else:
    inf=lb.n_node_features
    X=[a.x for a in alls]; EI=[a.edge_index for a in alls]; EW=[a.edge_attr for a in alls]
    Yt=[torch.tensor(Y[i],dtype=torch.long) for i in range(T-lags)]
    AC=[torch.tensor(active[i]) for i in range(T-lags)]
    torch.manual_seed(SEED)
    model={"gconvgru":GG,"static":SG,"evolveo":EO}[M](inf)
    opt=torch.optim.Adam(model.parameters(),lr=LR); rec=(M=="gconvgru")
    for _ in range(EP):
        h=None
        for st in range(0,n_tr,WIN):
            if rec and h is not None: h=h.detach()
            if M=="evolveo": model.reset()
            opt.zero_grad(); wl=0.0; c=0
            for i in range(st,min(st+WIN,n_tr)):
                if rec: lg,h=model(X[i],EI[i],EW[i],h)
                else: lg=model(X[i],EI[i],EW[i])
                m=AC[i]&trv
                if m.sum()==0: continue
                wl=wl+focal(lg[m],Yt[i][m],base._snapshot_class_weights(Yt[i][m])); c+=1
            if c:(wl/c).backward(); opt.step()
    model.eval(); P,L=[],[]; h=None
    with torch.no_grad():
        if M=="evolveo": model.reset()
        for i in range(T-lags):
            if rec: lg,h=model(X[i],EI[i],EW[i],h)
            else: lg=model(X[i],EI[i],EW[i])
            m=AC[i]&tev
            if m.sum()>0: P.append(torch.softmax(lg,1)[m,1].numpy()); L.append(Yt[i][m].numpy())
    p=np.concatenate(P); l=np.concatenate(L)

np.savez(f"roc_{M}.npz", prob=p, label=l)
from sklearn.metrics import roc_auc_score
print(f"{M}: saved roc_{M}.npz  ROC-AUC={roc_auc_score(l,p)*100:.2f}", flush=True)
