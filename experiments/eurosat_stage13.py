#!/usr/bin/env python3
"""Stage 13 - paper revision experiments (panel plan items 2,3,6,9):
(a) BASELINES: predict test-set coverage deficit via TV-bound vs cal-set direct
    shift vs accuracy-drop proxy; report signed error + |error|.
(b) ESTIMATORS: TV(1000 bins), TV(100 bins), KS, Wasserstein-1 as predictors.
(c) APS score: does the bound hold for APS as it does for LAC?
(d) sets_demo data: one EuroSAT tile's prediction sets at FP32/4bit/3bit.
2 datasets x ResNet18 x bits {8,4,3} x 3 seeds. Output stage13_results.json + sets_demo.png
"""
import json, os, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS=[0,1,2]; ALPHA=0.10; BITS=[8,4,3]; BATCH=256
EA=os.environ.get("EA_DIR",os.path.expanduser("~/eurosat-assured"))
ROOT=os.path.join(EA,"data"); dev="cuda"

def get_data(name,seed):
    g=torch.Generator().manual_seed(seed)
    if name=="eurosat":
        nrm=tv.transforms.Normalize([0.3444,0.3803,0.4078],[0.0914,0.0651,0.0552])
        aug=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),tv.transforms.RandomVerticalFlip(),tv.transforms.ToTensor(),nrm])
        ev=tv.transforms.Compose([tv.transforms.ToTensor(),nrm])
        full=tv.datasets.EuroSAT(root=ROOT,download=True); n=len(full)
        a,b=int(0.7*n),int(0.15*n)
        i1,i2,i3=random_split(range(n),[a,b,n-a-b],generator=g)
    else:
        nrm=tv.transforms.Normalize([0.4914,0.4822,0.4465],[0.247,0.243,0.261])
        aug=tv.transforms.Compose([tv.transforms.RandomCrop(32,4),tv.transforms.RandomHorizontalFlip(),tv.transforms.ToTensor(),nrm])
        ev=tv.transforms.Compose([tv.transforms.ToTensor(),nrm])
        tr=tv.datasets.CIFAR10(root=ROOT,train=True,download=True)
        te=tv.datasets.CIFAR10(root=ROOT,train=False,download=True)
        full=None
        a=int(0.85*len(tr)); i1,i2=random_split(range(len(tr)),[a,len(tr)-a],generator=g)
    class S(torch.utils.data.Dataset):
        def __init__(s,base,i,t): s.b,s.i,s.t=base,list(i),t
        def __len__(s): return len(s.i)
        def __getitem__(s,k):
            x,y=s.b[s.i[k]]; return s.t(x),y
    if name=="eurosat":
        return S(full,i1,aug),S(full,i2,ev),S(full,i3,ev)
    return S(tr,i1,aug),S(tr,i2,ev),S(te,range(len(te)),ev)

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
def probs(m,loader):
    m.eval(); ps,ys=[],[]
    for x,y in loader: ps.append(torch.softmax(m(x.to(dev)),1).cpu()); ys.append(y)
    return torch.cat(ps),torch.cat(ys)

def lac(p,y): return 1-p[torch.arange(len(y)),y]
def aps(p,y):
    srt,idx=p.sort(1,descending=True); cum=srt.cumsum(1)
    r=(idx==y.unsqueeze(1)).float().argmax(1)
    return cum[torch.arange(len(y)),r]
def qhat_of(s,a=ALPHA):
    k=min(int((1-a)*(len(s)+1))/len(s),1.0); return torch.quantile(s,k).item()
def tvh(a,b,nb):
    ha=torch.histc(a,nb,0,1)/len(a); hb=torch.histc(b,nb,0,1)/len(b)
    return 0.5*(ha-hb).abs().sum().item()
def ks(a,b):
    g=torch.linspace(0,1,2000)
    return ((a.unsqueeze(0)<=g.unsqueeze(1)).float().mean(1)-(b.unsqueeze(0)<=g.unsqueeze(1)).float().mean(1)).abs().max().item()
def w1(a,b):
    n=min(len(a),len(b))
    return (a.sort().values[:n]-b.sort().values[:n]).abs().mean().item()

rows=[]; aps_rows=[]; t0=time.time()
for ds in ["eurosat","cifar10"]:
    for seed in SEEDS:
        torch.manual_seed(seed)
        dtr,dca,dte=get_data(ds,seed)
        ltr=DataLoader(dtr,batch_size=BATCH,shuffle=True,num_workers=8,pin_memory=True)
        lca=DataLoader(dca,batch_size=BATCH,num_workers=4)
        lte=DataLoader(dte,batch_size=BATCH,num_workers=4)
        M=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); M.fc=nn.Linear(512,10); M=M.to(dev)
        o=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
        sc_=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=12)
        for e in range(12):
            M.train()
            for x,y in ltr:
                x,y=x.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(M(x),y).backward(); o.step()
            sc_.step()
        p_ca,y_ca=probs(M,lca); p_te,y_te=probs(M,lte)
        s_ca=lac(p_ca,y_ca); qh=qhat_of(s_ca)
        acc0=(p_te.argmax(1)==y_te).float().mean().item()
        sa_ca=aps(p_ca,y_ca); qh_aps=qhat_of(sa_ca)
        for bits in BITS:
            Q=fq(M,bits)
            pq_ca,_=probs(Q,lca); pq_te,_=probs(Q,lte)
            sq_ca=lac(pq_ca,y_ca); sq_te=lac(pq_te,y_te)
            true_def=(1-ALPHA)-(sq_te<=qh).float().mean().item()
            pred_tv=tvh(s_ca,sq_ca,1000)                     # TV bound from CAL only
            pred_calshift=(1-ALPHA)-(sq_ca<=qh).float().mean().item()  # direct cal-set deficit
            pred_accdrop=acc0-(pq_te.argmax(1)==y_te).float().mean().item() # acc-drop proxy (uses test acc; also cal version)
            pred_accdrop_cal=( (p_ca.argmax(1)==y_ca).float().mean() - (pq_ca.argmax(1)==y_ca).float().mean() ).item()
            rows.append(dict(ds=ds,seed=seed,bits=bits,true_deficit=true_def,
                tv1000=pred_tv,tv100=tvh(s_ca,sq_ca,100),ks=ks(s_ca,sq_ca),w1=w1(s_ca,sq_ca),
                cal_shift=pred_calshift,acc_drop_cal=pred_accdrop_cal,acc_drop_test=pred_accdrop))
            saq_te=aps(pq_te,y_te)
            aps_def=(1-ALPHA)-(saq_te<=qh_aps).float().mean().item()
            aps_tv=tvh(sa_ca,aps(pq_ca,y_ca),1000)
            aps_rows.append(dict(ds=ds,seed=seed,bits=bits,aps_deficit=aps_def,aps_tv=aps_tv,
                holds=bool(aps_def<=aps_tv+1e-9)))
        # sets demo (eurosat seed0): one tile predictions at FP32/4/3
        if ds=="eurosat" and seed==0:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            CL=["AnnualCrop","Forest","Herb.","Highway","Industrial","Pasture","PermCrop","Residential","River","SeaLake"]
            x0,y0=dte[7]
            raw=dte.b[dte.i[7]][0]
            fig,axes=plt.subplots(1,3,figsize=(8.6,3.3),dpi=150)
            for ax,bb in zip(axes,[None,4,3]):
                m2=M if bb is None else fq(M,bb)
                with torch.no_grad():
                    pp=torch.softmax(m2(x0.unsqueeze(0).to(dev)),1)[0].cpu()
                q2=qh if bb is None else qhat_of(lac(*probs(fq(M,bb),lca)[0:1]+ (probs(fq(M,bb),lca)[1],)))
                st=[CL[i] for i in range(10) if pp[i]>=1-qh]
                ax.imshow(raw); ax.axis("off")
                ax.set_title(("FP32" if bb is None else f"{bb}-bit")+f" set (stale cal):\n{{{', '.join(st) if st else 'EMPTY (refuse)'}}}",fontsize=8)
            fig.suptitle(f"One real tile (true: {CL[y0]}): prediction sets as precision drops",fontsize=10)
            fig.tight_layout(); fig.savefig(os.path.join(EA,"sets_demo.png"),bbox_inches="tight")
        print(f"[s13] {ds}/s{seed} done ({time.time()-t0:.0f}s)",flush=True)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std())}
base={}
for ds in ["eurosat","cifar10"]:
    for bits in BITS:
        sel=[r for r in rows if r["ds"]==ds and r["bits"]==bits]
        base[f"{ds}/{bits}"]={k:ms([r[k] for r in sel]) for k in
            ["true_deficit","tv1000","tv100","ks","w1","cal_shift","acc_drop_cal","acc_drop_test"]}
apss={}
for ds in ["eurosat","cifar10"]:
    for bits in BITS:
        sel=[r for r in aps_rows if r["ds"]==ds and r["bits"]==bits]
        apss[f"{ds}/{bits}"]={"aps_deficit":ms([r["aps_deficit"] for r in sel]),
            "aps_tv":ms([r["aps_tv"] for r in sel]),
            "holds_all":all(r["holds"] for r in sel)}
summary={"n_seeds":len(SEEDS),"alpha":ALPHA,"baselines":base,"aps":apss,
    "runtime_s":round(time.time()-t0,1)}
with open(os.path.join(EA,"stage13_results.json"),"w") as f:
    json.dump({"summary":summary,"rows":rows,"aps_rows":aps_rows},f,indent=2)
print(f"[s13] DONE ({summary['runtime_s']}s)",flush=True)
