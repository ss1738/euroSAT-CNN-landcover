#!/usr/bin/env python3
"""Stage 8 - quantization-vs-conformal scaling law. Fake-quant weights to
k bits (8/6/4/3/2), measure coverage (stale vs recal), accuracy, weight and
logit perturbation. Output: ~/eurosat-assured/stage8_results.json"""
import json, os, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS=[0,1,2,3,4]; ALPHA=0.10; EPOCHS=12; BATCH=256; BITS=[8,6,4,3,2]
ROOT=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data"); OUT=os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")); dev="cuda"
MEAN,STD=[0.3444,0.3803,0.4078],[0.0914,0.0651,0.0552]
norm=tv.transforms.Normalize(MEAN,STD)
tf_tr=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),tv.transforms.RandomVerticalFlip(),tv.transforms.ToTensor(),norm])
tf_ev=tv.transforms.Compose([tv.transforms.ToTensor(),norm])
full=tv.datasets.EuroSAT(root=ROOT,download=True); n=len(full)
n_tr,n_ca=int(0.7*n),int(0.15*n); n_te=n-n_tr-n_ca

class Sub(torch.utils.data.Dataset):
    def __init__(s,b,i,t): s.b,s.i,s.t=b,list(i),t
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        img,y=s.b[s.i[k]]; return s.t(img),y

def fake_quant(model,bits):
    m=copy.deepcopy(model)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod,(nn.Conv2d,nn.Linear)):
                w=mod.weight.data
                qmax=2**(bits-1)-1
                scale=w.abs().amax(dim=tuple(range(1,w.dim())),keepdim=True).clamp(min=1e-8)/qmax
                mod.weight.data=torch.round(w/scale).clamp(-qmax-1,qmax)*scale
    return m

@torch.no_grad()
def sm(m,loader):
    m.eval(); ps,ys=[],[]
    for x,y in loader: ps.append(torch.softmax(m(x.to(dev)),1).cpu()); ys.append(y)
    return torch.cat(ps),torch.cat(ys)

def qhat_of(p,y,a):
    s=1-p[torch.arange(len(y)),y]; k=min(int((1-a)*(len(s)+1))/len(s),1.0)
    return torch.quantile(s,k).item()
def cov_of(p,y,qh): return ((p>=(1-qh))[torch.arange(len(y)),y]).float().mean().item()

rows=[]; t0=time.time()
for seed in SEEDS:
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    i_tr,i_ca,i_te=random_split(range(n),[n_tr,n_ca,n_te],generator=g)
    l_tr=DataLoader(Sub(full,i_tr,tf_tr),batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True)
    l_ca=DataLoader(Sub(full,i_ca,tf_ev),batch_size=BATCH,num_workers=4)
    l_te=DataLoader(Sub(full,i_te,tf_ev),batch_size=BATCH,num_workers=4)
    M=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); M.fc=nn.Linear(512,10); M=M.to(dev)
    o=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=EPOCHS)
    for e in range(EPOCHS):
        M.train()
        for x,y in l_tr:
            x,y=x.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(M(x),y).backward(); o.step()
        sc.step()
    p_ca,y_ca=sm(M,l_ca); p_te,y_te=sm(M,l_te)
    qh_fp=qhat_of(p_ca,y_ca,ALPHA)
    w_fp=torch.cat([p.data.flatten().cpu() for p in M.parameters()])
    for bits in BITS:
        Q=fake_quant(M,bits)
        w_q=torch.cat([p.data.flatten().cpu() for p in Q.parameters()])
        wpert=(w_q-w_fp).norm().item()/w_fp.norm().item()
        pq_ca,_=sm(Q,l_ca); pq_te,_=sm(Q,l_te)
        lshift=(pq_te-p_te).abs().mean().item()
        rows.append(dict(seed=seed,bits=bits,
            acc=(pq_te.argmax(1)==y_te).float().mean().item(),
            cov_stale=cov_of(pq_te,y_te,qh_fp),
            cov_recal=cov_of(pq_te,y_te,qhat_of(pq_ca,y_ca,ALPHA)),
            w_pert_rel=wpert, prob_shift=lshift))
    rows.append(dict(seed=seed,bits=32,acc=(p_te.argmax(1)==y_te).float().mean().item(),
        cov_stale=cov_of(p_te,y_te,qh_fp),cov_recal=cov_of(p_te,y_te,qh_fp),
        w_pert_rel=0.0,prob_shift=0.0))
    print(f"[s8] seed {seed} done ({time.time()-t0:.0f}s)",flush=True)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std())}
by_bits={str(b):{k:ms([r[k] for r in rows if r["bits"]==b])
    for k in ["acc","cov_stale","cov_recal","w_pert_rel","prob_shift"]} for b in [32]+BITS}
summary={"n_seeds":len(SEEDS),"alpha":ALPHA,"target":1-ALPHA,"by_bits":by_bits,
         "runtime_s":round(time.time()-t0,1)}
with open(f"{OUT}/stage8_results.json","w") as f:
    json.dump({"summary":summary,"rows":rows},f,indent=2)
print(f"[s8] DONE ({summary['runtime_s']}s)",flush=True)
