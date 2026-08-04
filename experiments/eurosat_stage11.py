#!/usr/bin/env python3
"""Stage 11 - fault x precision factorial. SEU flips injected into the QUANTIZED
integer weight codes (faithful stored-weight upset) at bits in {8,4,3} plus FP32
float flips, k in {1,10,100}, 8 trials. Measures: accuracy, gate accept/err_accepted
(conformal+energy, clean-cal thresholds), output-silent fraction (k=1: argmax
unchanged on 1024-image probe). Also weight-hash overhead: sha256 over weights vs
one-batch inference time. Output stage11_results.json"""
import json, os, time, copy, random, hashlib
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS=[0,1,2]; ALPHA=0.10; BATCH=256; KS=[1,10,100]; TRIALS=8
ROOT=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data"); OUT=os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")); dev="cuda"
norm=tv.transforms.Normalize([0.3444,0.3803,0.4078],[0.0914,0.0651,0.0552])
tf_tr=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),tv.transforms.RandomVerticalFlip(),tv.transforms.ToTensor(),norm])
tf_ev=tv.transforms.Compose([tv.transforms.ToTensor(),norm])
full=tv.datasets.EuroSAT(root=ROOT,download=True); n=len(full)
n_tr,n_ca=int(0.7*n),int(0.15*n); n_te=n-n_tr-n_ca

class Sub(torch.utils.data.Dataset):
    def __init__(s,i,t): s.i,s.t=list(i),t
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        x,y=full[s.i[k]]; return s.t(x),y

def quant_int(model,bits):
    """Quantize weights; return model + list of (module, scale, qmax)."""
    m=copy.deepcopy(model); meta=[]
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod,(nn.Conv2d,nn.Linear)):
                w=mod.weight.data; qm=2**(bits-1)-1
                sc=w.abs().amax(dim=tuple(range(1,w.dim())),keepdim=True).clamp(min=1e-8)/qm
                mod.weight.data=torch.round(w/sc).clamp(-qm-1,qm)*sc
                meta.append((mod,sc,qm))
    return m,meta

def flip_int(m_meta,k,rng,bits):
    m,meta=m_meta; m=copy.deepcopy(m)
    mods=[mo for mo in m.modules() if isinstance(mo,(nn.Conv2d,nn.Linear))]
    metas=[]
    with torch.no_grad():
        for mo in mods:
            w=mo.weight.data; qm=2**(bits-1)-1
            sc=w.abs().amax(dim=tuple(range(1,w.dim())),keepdim=True).clamp(min=1e-8)/qm
            metas.append((mo,sc,qm))
        sizes=[mo.weight.numel() for mo,_,_ in metas]; total=sum(sizes)
        for _ in range(k):
            gi=rng.randrange(total)
            for (mo,sc,qm),sz in zip(metas,sizes):
                if gi<sz:
                    fl=mo.weight.data.view(-1)
                    scf=sc.expand_as(mo.weight.data).contiguous().view(-1)
                    code=int(round((fl[gi]/scf[gi]).item()))
                    bit=rng.randrange(bits)
                    code=(code & (2**bits-1)) ^ (1<<bit)
                    if code>=2**(bits-1): code-=2**bits
                    fl[gi]=max(-qm-1,min(qm,code))*scf[gi]
                    break
                gi-=sz
    return m

def flip_fp32(model,k,rng):
    m=copy.deepcopy(model)
    ps=[p for p in m.parameters()]
    sizes=[p.numel() for p in ps]; total=sum(sizes)
    with torch.no_grad():
        for _ in range(k):
            gi=rng.randrange(total)
            for p,sz in zip(ps,sizes):
                if gi<sz:
                    v=p.data.view(-1)[gi:gi+1].view(torch.int32)
                    v^=(1<<rng.randrange(32)); break
                gi-=sz
    return m

@torch.no_grad()
def sm(m,loader):
    m=m.to(dev); m.eval(); ps,ys=[],[]
    for x,y in loader: ps.append(torch.softmax(m(x.to(dev)),1).cpu()); ys.append(y)
    return torch.cat(ps),torch.cat(ys)

def qhat_of(p,y):
    s=1-p[torch.arange(len(y)),y]
    k=min(int((1-ALPHA)*(len(s)+1))/len(s),1.0); return torch.quantile(s,k).item()
def energy(p): return -torch.log(p.clamp(min=1e-12)).min(1).values*0  # placeholder unused

rows=[]; silent_rows=[]; t0=time.time()
for seed in SEEDS:
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    i_tr,i_ca,i_te=random_split(range(n),[n_tr,n_ca,n_te],generator=g)
    l_tr=DataLoader(Sub(i_tr,tf_tr),batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True)
    l_ca=DataLoader(Sub(i_ca,tf_ev),batch_size=BATCH,num_workers=4)
    i_te=list(i_te); l_te=DataLoader(Sub(i_te[:2048],tf_ev),batch_size=BATCH,num_workers=4)
    M=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); M.fc=nn.Linear(512,10); M=M.to(dev)
    o=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
    sc_=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=12)
    for e in range(12):
        M.train()
        for x,y in l_tr:
            x,y=x.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(M(x),y).backward(); o.step()
        sc_.step()
    M=M.cpu()
    for bits in [32,8,4,3]:
        base=(copy.deepcopy(M) if bits==32 else quant_int(M,bits)[0])
        p_ca,y_ca=sm(copy.deepcopy(base),l_ca); qh=qhat_of(p_ca,y_ca)
        p0,y_te=sm(copy.deepcopy(base),l_te); pred0=p0.argmax(1)
        for k in KS:
            for tr in range(TRIALS):
                rng=random.Random(seed*10000+bits*100+k*10+tr)
                mf=flip_fp32(base,k,rng) if bits==32 else flip_int((base,None),k,rng,bits)
                p,y=sm(mf,l_te)
                ne=(p>=(1-qh)).sum(1)>0
                fin=torch.isfinite(p).all(1)
                acp=ne&fin
                wr=p.argmax(1)!=y
                rows.append(dict(seed=seed,bits=bits,k=k,trial=tr,
                    acc=(~wr).float().mean().item(),
                    accept=acp.float().mean().item(),
                    err_acc=(wr&acp).sum().item()/max(acp.sum().item(),1)))
                if k==1:
                    silent_rows.append(dict(seed=seed,bits=bits,
                        silent=bool((p.argmax(1)==pred0).all().item())))
        print(f"[s11] seed {seed} bits {bits} done ({time.time()-t0:.0f}s)",flush=True)
    # hash overhead (once, seed 0)
    if seed==0:
        wbytes=b"".join(p.data.numpy().tobytes() for p in M.parameters())
        t1=time.time(); h=hashlib.sha256(wbytes).hexdigest(); th=time.time()-t1
        x=torch.randn(1,3,64,64)
        Mc=copy.deepcopy(M).eval()
        t1=time.time()
        with torch.no_grad():
            for _ in range(10): Mc(x)
        ti=(time.time()-t1)/10
        hash_info=dict(hash_s=th,infer_s=ti,ratio=th/ti,weight_mb=len(wbytes)/1e6)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std())}
by={}
for bits in [32,8,4,3]:
    by[str(bits)]={}
    for k in KS:
        sel=[r for r in rows if r["bits"]==bits and r["k"]==k]
        by[str(bits)][str(k)]={kk:ms([r[kk] for r in sel]) for kk in ["acc","accept","err_acc"]}
silent={str(b):sum(r["silent"] for r in silent_rows if r["bits"]==b)/max(sum(1 for r in silent_rows if r["bits"]==b),1)
        for b in [32,8,4,3]}
summary={"n_seeds":len(SEEDS),"trials":TRIALS,"by_bits_k":by,
         "silent_fraction_k1":silent,"hash_overhead":hash_info,
         "runtime_s":round(time.time()-t0,1)}
with open(f"{OUT}/stage11_results.json","w") as f:
    json.dump({"summary":summary,"rows":rows,"silent_rows":silent_rows},f,indent=2)
print(f"[s11] DONE ({summary['runtime_s']}s)",flush=True)
