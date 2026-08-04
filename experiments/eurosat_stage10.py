#!/usr/bin/env python3
"""Stage 10 - generality of the TV-bound result: 2 datasets x 2 archs x 5 bits x 3 seeds,
plus set-size efficiency (stale vs recal). Output stage10_results.json"""
import json, os, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS=[0,1,2]; ALPHA=0.10; BATCH=256; BITS=[8,6,4,3,2]; NB=1000
ROOT=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data"); OUT=os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")); dev="cuda"

def get_data(name,seed):
    g=torch.Generator().manual_seed(seed)
    if name=="eurosat":
        nrm=tv.transforms.Normalize([0.3444,0.3803,0.4078],[0.0914,0.0651,0.0552])
        aug=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),tv.transforms.RandomVerticalFlip(),tv.transforms.ToTensor(),nrm])
        ev=tv.transforms.Compose([tv.transforms.ToTensor(),nrm])
        full=tv.datasets.EuroSAT(root=ROOT,download=True); n=len(full)
        a,b=int(0.7*n),int(0.15*n)
        i1,i2,i3=random_split(range(n),[a,b,n-a-b],generator=g)
        class S(torch.utils.data.Dataset):
            def __init__(s,i,t): s.i,s.t=list(i),t
            def __len__(s): return len(s.i)
            def __getitem__(s,k):
                x,y=full[s.i[k]]; return s.t(x),y
        return S(i1,aug),S(i2,ev),S(i3,ev)
    else:
        nrm=tv.transforms.Normalize([0.4914,0.4822,0.4465],[0.247,0.243,0.261])
        aug=tv.transforms.Compose([tv.transforms.RandomCrop(32,4),tv.transforms.RandomHorizontalFlip(),tv.transforms.ToTensor(),nrm])
        ev=tv.transforms.Compose([tv.transforms.ToTensor(),nrm])
        tr=tv.datasets.CIFAR10(root=ROOT,train=True,download=True)
        te=tv.datasets.CIFAR10(root=ROOT,train=False,download=True)
        a=int(0.85*len(tr)); i1,i2=random_split(range(len(tr)),[a,len(tr)-a],generator=g)
        class S(torch.utils.data.Dataset):
            def __init__(s,base,i,t): s.b,s.i,s.t=base,list(i),t
            def __len__(s): return len(s.i)
            def __getitem__(s,k):
                x,y=s.b[s.i[k]]; return s.t(x),y
        return S(tr,i1,aug),S(tr,i2,ev),S(te,range(len(te)),ev)

def get_model(arch):
    if arch=="resnet18":
        m=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); m.fc=nn.Linear(512,10)
    else:
        m=tv.models.mobilenet_v3_small(weights=tv.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3]=nn.Linear(1024,10)
    return m

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

def sc_of(p,y): return 1-p[torch.arange(len(y)),y]
def qhat_of(s):
    k=min(int((1-ALPHA)*(len(s)+1))/len(s),1.0); return torch.quantile(s,k).item()
def tv_hist(a,b):
    ha=torch.histc(a,NB,0,1)/len(a); hb=torch.histc(b,NB,0,1)/len(b)
    return 0.5*(ha-hb).abs().sum().item()
def setsize(p,qh): return (p>=(1-qh)).sum(1).float().mean().item()

rows=[]; t0=time.time()
for ds in ["eurosat","cifar10"]:
    for arch in ["resnet18","mobilenetv3s"]:
        for seed in SEEDS:
            torch.manual_seed(seed)
            dtr,dca,dte=get_data(ds,seed)
            ltr=DataLoader(dtr,batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True)
            lca=DataLoader(dca,batch_size=BATCH,num_workers=4)
            lte=DataLoader(dte,batch_size=BATCH,num_workers=4)
            M=get_model(arch).to(dev)
            o=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
            s_=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=12)
            for e in range(12):
                M.train()
                for x,y in ltr:
                    x,y=x.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(M(x),y).backward(); o.step()
                s_.step()
            p_ca,y_ca=sm(M,lca); p_te,y_te=sm(M,lte)
            s_ca=sc_of(p_ca,y_ca); qh=qhat_of(s_ca)
            base_acc=(p_te.argmax(1)==y_te).float().mean().item()
            for bits in BITS:
                Q=fq(M,bits)
                pq_te,_=sm(Q,lte); pq_ca,_=sm(Q,lca)
                sq_te=sc_of(pq_te,y_te); sq_ca=sc_of(pq_ca,y_ca)
                cov=(sq_te<=qh).float().mean().item(); tvd=tv_hist(s_ca,sq_te)
                qh_r=qhat_of(sq_ca)
                rows.append(dict(dataset=ds,arch=arch,seed=seed,bits=bits,base_acc=base_acc,
                    acc=(pq_te.argmax(1)==y_te).float().mean().item(),
                    cov_stale=cov,deficit=(1-ALPHA)-cov,tv=tvd,
                    bound_holds=bool(cov>=1-ALPHA-tvd-1e-9),
                    cov_recal=(sq_te<=qh_r).float().mean().item(),
                    size_stale=setsize(pq_te,qh),size_recal=setsize(pq_te,qh_r)))
            print(f"[s10] {ds}/{arch}/s{seed} done ({time.time()-t0:.0f}s)",flush=True)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std())}
by={}
for ds in ["eurosat","cifar10"]:
    for arch in ["resnet18","mobilenetv3s"]:
        key=f"{ds}/{arch}"; by[key]={}
        for b in BITS:
            sel=[r for r in rows if r["dataset"]==ds and r["arch"]==arch and r["bits"]==b]
            by[key][str(b)]={k:ms([r[k] for r in sel]) for k in
                ["deficit","tv","cov_stale","cov_recal","size_recal","acc"]}
holds=all(r["bound_holds"] for r in rows)
frac=sum(r["bound_holds"] for r in rows)/len(rows)
summary={"n_seeds":len(SEEDS),"alpha":ALPHA,"grid":by,"bound_holds_all":holds,
         "bound_holds_frac":frac,"runtime_s":round(time.time()-t0,1)}
with open(f"{OUT}/stage10_results.json","w") as f:
    json.dump({"summary":summary,"rows":rows},f,indent=2)
print(f"[s10] DONE holds={holds} frac={frac:.2f} ({summary['runtime_s']}s)",flush=True)
