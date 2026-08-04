#!/usr/bin/env python3
"""Stage 7 - multispectral EuroSAT: 13-band baseline + gate, band-selection curve,
MS student. Output: ~/eurosat-assured/stage7_results.json"""
import json, os, time, glob, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

SEEDS=[0,1,2]; ALPHA=0.10; EP_T=12; EP_S=40; BATCH=256
ROOT=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data/ms/EuroSAT_MS")
OUT=os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")); dev="cuda"
CACHE=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data/ms_cache.npz")
SUBSETS={"rgb3":[3,2,1],"nir5":[1,2,3,7,10],"b8":[1,2,3,4,5,6,7,10],"all13":list(range(13))}

if os.path.exists(CACHE):
    z=np.load(CACHE); X=z["X"]; Y=z["Y"]
else:
    import tifffile
    classes=sorted(os.listdir(ROOT)); xs=[]; ys=[]
    for ci,c in enumerate(classes):
        for f in sorted(glob.glob(f"{ROOT}/{c}/*.tif")):
            xs.append(tifffile.imread(f)); ys.append(ci)
    X=np.stack(xs).astype(np.uint16); Y=np.array(ys,dtype=np.int64)
    np.savez_compressed(CACHE,X=X,Y=Y)
print(f"[s7] data {X.shape}",flush=True)
Xt=torch.from_numpy(X.astype(np.float32)/10000.0).permute(0,3,1,2).contiguous()
Yt=torch.from_numpy(Y); n=len(Yt); n_tr,n_ca=int(0.7*n),int(0.15*n); n_te=n-n_tr-n_ca

class DS(Dataset):
    def __init__(s,idx,bands,noise=False,mu=None,sd=None):
        s.idx=list(idx); s.b=bands; s.noise=noise; s.mu=mu; s.sd=sd
    def __len__(s): return len(s.idx)
    def __getitem__(s,i):
        x=Xt[s.idx[i]][s.b].clone(); y=int(Yt[s.idx[i]])
        if s.noise: x=(x+0.05*torch.randn_like(x)).clamp(0,3)
        raw=x.clone(); xn=(x-s.mu)/s.sd
        return xn,raw,y

def stats_of(idx,bands):
    xs=Xt[idx][:, bands]; mu=xs.mean(dim=(0,2,3),keepdim=True)[0]; sd=xs.std(dim=(0,2,3),keepdim=True)[0]
    return mu,sd

class DSB(nn.Module):
    def __init__(s,ci,co,st):
        super().__init__(); s.dw=nn.Conv2d(ci,ci,3,st,1,groups=ci,bias=False); s.b1=nn.BatchNorm2d(ci)
        s.pw=nn.Conv2d(ci,co,1,bias=False); s.b2=nn.BatchNorm2d(co)
    def forward(s,x): return F.relu(s.b2(s.pw(F.relu(s.b1(s.dw(x))))))
class Tiny(nn.Module):
    def __init__(s,cin,ncls=10):
        super().__init__(); s.stem=nn.Sequential(nn.Conv2d(cin,48,3,2,1,bias=False),nn.BatchNorm2d(48),nn.ReLU())
        s.b1=DSB(48,96,2); s.b2=DSB(96,192,2); s.b3=DSB(192,192,2); s.fc=nn.Linear(192,ncls)
    def forward(s,x): return s.fc(F.adaptive_avg_pool2d(s.b3(s.b2(s.b1(s.stem(x)))),1).flatten(1))

import torchvision as tv
def resnet(cin):
    m=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    w=m.conv1.weight.data; m.conv1=nn.Conv2d(cin,64,7,2,3,bias=False)
    m.conv1.weight.data=w.mean(1,keepdim=True).repeat(1,cin,1,1)*(3.0/cin); m.fc=nn.Linear(512,10)
    return m

@torch.no_grad()
def ev(m,loader):
    m.eval(); ls,hf,sd,ys=[],[],[],[]
    for xn,raw,y in loader:
        ls.append(m(xn.to(dev)).cpu())
        hf.append((raw-F.avg_pool2d(raw,3,1,1)).abs().mean(dim=(1,2,3))); sd.append(raw.std(dim=(2,3)).mean(1)); ys.append(y)
    return torch.cat(ls),torch.cat(hf),torch.cat(sd),torch.cat(ys)

def train(m,loader,epochs,lr):
    m=m.to(dev); o=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-4)
    s=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs)
    for e in range(epochs):
        m.train()
        for xn,_,y in loader:
            xn,y=xn.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(m(xn),y).backward(); o.step()
        s.step()
    return m

def qhat_of(p,y,a):
    sc=1-p[torch.arange(len(y)),y]; k=min(int((1-a)*(len(sc)+1))/len(sc),1.0)
    return torch.quantile(sc,k).item()
def energy(l): return -torch.logsumexp(l,1)

rows=[]; band_rows=[]; t0=time.time()
for seed in SEEDS:
    g=torch.Generator().manual_seed(seed)
    i_tr,i_ca,i_te=random_split(range(n),[n_tr,n_ca,n_te],generator=g)
    i_tr,i_ca,i_te=list(i_tr),list(i_ca),list(i_te)
    for name,bands in SUBSETS.items():
        mu,sd_=stats_of(i_tr,bands)
        l_tr=DataLoader(DS(i_tr,bands,mu=mu,sd=sd_),batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True)
        l_ca=DataLoader(DS(i_ca,bands,mu=mu,sd=sd_),batch_size=BATCH,num_workers=4)
        l_te=DataLoader(DS(i_te,bands,mu=mu,sd=sd_),batch_size=BATCH,num_workers=4)
        l_no=DataLoader(DS(i_te,bands,noise=True,mu=mu,sd=sd_),batch_size=BATCH,num_workers=4)
        S=train(Tiny(len(bands)),l_tr,EP_S,1e-3)
        lg,_,_,y=ev(S,l_te); acc=(lg.argmax(1)==y).float().mean().item()
        band_rows.append(dict(seed=seed,subset=name,nbands=len(bands),acc=acc,
                              bytes_per_tile=len(bands)*64*64*2))
        if name=="all13":
            T=train(resnet(13),l_tr,EP_T,3e-4)
            for mn,m in [("resnet13",T),("tiny13",S)]:
                lg_ca,hf_ca,sd_ca,y_ca=ev(m,l_ca); p_ca=torch.softmax(lg_ca,1)
                qh=qhat_of(p_ca,y_ca,ALPHA); et=torch.quantile(energy(lg_ca),0.95).item()
                env=[torch.quantile(hf_ca,0.005).item(),torch.quantile(hf_ca,0.995).item(),
                     torch.quantile(sd_ca,0.005).item(),torch.quantile(sd_ca,0.995).item()]
                for kind,ld in [("clean",l_te),("noise",l_no)]:
                    lg,hf,sd2,y=ev(m,ld); p=torch.softmax(lg,1)
                    ne=(p>=(1-qh)).sum(1)>0; e=energy(lg); eo=torch.isfinite(e)&(e<=et)
                    io=(hf>=env[0])&(hf<=env[1])&(sd2>=env[2])&(sd2<=env[3])
                    acp=ne&eo&io; wr=p.argmax(1)!=y
                    rows.append(dict(seed=seed,model=mn,corruption=kind,
                        acc=(~wr).float().mean().item(),accept_rate=acp.float().mean().item(),
                        cov=( (p>=(1-qh))[torch.arange(len(y)),y] ).float().mean().item(),
                        err_accepted=(wr&acp).sum().item()/max(acp.sum().item(),1)))
    print(f"[s7] seed {seed} done ({time.time()-t0:.0f}s)",flush=True)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std()) if len(v)>1 else 0.0}
gate={mn:{k:{kk:ms([r[kk] for r in rows if r["model"]==mn and r["corruption"]==k])
      for kk in ["acc","accept_rate","cov","err_accepted"]} for k in ["clean","noise"]}
      for mn in ["resnet13","tiny13"]}
bands={nm:{ "nbands":[r["nbands"] for r in band_rows if r["subset"]==nm][0],
            "bytes_per_tile":[r["bytes_per_tile"] for r in band_rows if r["subset"]==nm][0],
            "acc":ms([r["acc"] for r in band_rows if r["subset"]==nm])} for nm in SUBSETS}
summary={"n_seeds":len(SEEDS),"alpha":ALPHA,"gate_ms":gate,"band_curve":bands,
         "runtime_s":round(time.time()-t0,1)}
with open(f"{OUT}/stage7_results.json","w") as f:
    json.dump({"summary":summary,"rows":rows,"band_rows":band_rows},f,indent=2)
print(f"[s7] DONE ({summary['runtime_s']}s)",flush=True)
