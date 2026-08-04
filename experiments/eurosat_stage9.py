#!/usr/bin/env python3
"""Stage 9 - test the Barber-style bound: coverage_stale >= (1-a) - TV(scores_fp32, scores_quant).
Estimate TV via 200-bin histograms + KS statistic. Output stage9_results.json"""
import json, os, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS=[0,1,2,3,4]; ALPHA=0.10; EPOCHS=12; BATCH=256; BITS=[8,6,4,3,2]; NB=200
ROOT=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data"); OUT=os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")); dev="cuda"
norm=tv.transforms.Normalize([0.3444,0.3803,0.4078],[0.0914,0.0651,0.0552])
tf_tr=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),tv.transforms.RandomVerticalFlip(),tv.transforms.ToTensor(),norm])
tf_ev=tv.transforms.Compose([tv.transforms.ToTensor(),norm])
full=tv.datasets.EuroSAT(root=ROOT,download=True); n=len(full)
n_tr,n_ca=int(0.7*n),int(0.15*n); n_te=n-n_tr-n_ca

class Sub(torch.utils.data.Dataset):
    def __init__(s,b,i,t): s.b,s.i,s.t=b,list(i),t
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        img,y=s.b[s.i[k]]; return s.t(img),y

def fq(model,bits):
    m=copy.deepcopy(model)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod,(nn.Conv2d,nn.Linear)):
                w=mod.weight.data; qm=2**(bits-1)-1
                sc=w.abs().amax(dim=tuple(range(1,w.dim())),keepdim=True).clamp(min=1e-8)/qm
                mod.weight.data=torch.round(w/sc).clamp(-qm-1,qm)*sc
    return m

@torch.no_grad()
def sm(m,loader):
    m.eval(); ps,ys=[],[]
    for x,y in loader: ps.append(torch.softmax(m(x.to(dev)),1).cpu()); ys.append(y)
    return torch.cat(ps),torch.cat(ys)

def scores(p,y): return 1-p[torch.arange(len(y)),y]
def qhat_of(s):
    k=min(int((1-ALPHA)*(len(s)+1))/len(s),1.0); return torch.quantile(s,k).item()
def tv_hist(a,b):
    ha=torch.histc(a,NB,0,1)/len(a); hb=torch.histc(b,NB,0,1)/len(b)
    return 0.5*(ha-hb).abs().sum().item()
def ks(a,b):
    grid=torch.linspace(0,1,400)
    ca=(a.unsqueeze(0)<=grid.unsqueeze(1)).float().mean(1)
    cb=(b.unsqueeze(0)<=grid.unsqueeze(1)).float().mean(1)
    return (ca-cb).abs().max().item()

rows=[]; t0=time.time()
for seed in SEEDS:
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    i_tr,i_ca,i_te=random_split(range(n),[n_tr,n_ca,n_te],generator=g)
    l_tr=DataLoader(Sub(full,i_tr,tf_tr),batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True)
    l_ca=DataLoader(Sub(full,i_ca,tf_ev),batch_size=BATCH,num_workers=4)
    l_te=DataLoader(Sub(full,i_te,tf_ev),batch_size=BATCH,num_workers=4)
    M=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); M.fc=nn.Linear(512,10); M=M.to(dev)
    o=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
    sc_=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=EPOCHS)
    for e in range(EPOCHS):
        M.train()
        for x,y in l_tr:
            x,y=x.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(M(x),y).backward(); o.step()
        sc_.step()
    p_ca,y_ca=sm(M,l_ca); p_te,y_te=sm(M,l_te)
    s_ca=scores(p_ca,y_ca); qh=qhat_of(s_ca)
    for bits in BITS:
        Q=fq(M,bits)
        pq_te,_=sm(Q,l_te); pq_ca,_=sm(Q,l_ca)
        sq_te=scores(pq_te,y_te); sq_ca=scores(pq_ca,y_ca)
        cov=(sq_te<=qh).float().mean().item()
        deficit=(1-ALPHA)-cov
        tv1=tv_hist(s_ca,sq_te); k1=ks(s_ca,sq_te); tv_cal=tv_hist(s_ca,sq_ca)
        rows.append(dict(seed=seed,bits=bits,cov_stale=cov,deficit=deficit,
            tv=tv1,ks=k1,tv_cal_only=tv_cal,
            bound_lo=1-ALPHA-tv1,bound_holds=bool(cov>=1-ALPHA-tv1-1e-9)))
    print(f"[s9] seed {seed} done ({time.time()-t0:.0f}s)",flush=True)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std())}
by={str(b):{k:ms([r[k] for r in rows if r["bits"]==b]) for k in
    ["cov_stale","deficit","tv","ks","bound_lo"]} for b in BITS}
holds=all(r["bound_holds"] for r in rows)
summary={"n_seeds":len(SEEDS),"alpha":ALPHA,"by_bits":by,"bound_holds_all_runs":holds,
         "runtime_s":round(time.time()-t0,1)}
with open(f"{OUT}/stage9_results.json","w") as f:
    json.dump({"summary":summary,"rows":rows},f,indent=2)
print(f"[s9] DONE bound_holds_all={holds} ({summary['runtime_s']}s)",flush=True)
