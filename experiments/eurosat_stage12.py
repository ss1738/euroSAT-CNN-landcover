#!/usr/bin/env python3
"""Stage 12 - mission-lifetime calibration drift: 'recalibrate the guarantee,
not the model'. Simulates 24 weeks of progressive sensor degradation
(noise + brightness drift), measures conformal coverage under:
  A) launch calibration, never refreshed
  B) score refresh every 4 weeks (200 labeled samples, ~KBs of scores)
  C) score refresh every 12 weeks
No retraining anywhere. Output stage12_results.json"""
import json, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS=[0,1,2]; ALPHA=0.10; WEEKS=24; PER_WEEK=500; REFRESH_N=200
ROOT=os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data"); OUT=os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")); dev="cuda"
norm=tv.transforms.Normalize([0.3444,0.3803,0.4078],[0.0914,0.0651,0.0552])
tf_tr=tv.transforms.Compose([tv.transforms.RandomHorizontalFlip(),tv.transforms.RandomVerticalFlip(),tv.transforms.ToTensor(),norm])
full=tv.datasets.EuroSAT(root=ROOT,download=True); n=len(full)

def drift_level(week):      # progressive degradation profile
    f=week/WEEKS
    return dict(noise=0.05*f, bright=1.0+0.35*f)

def degrade(x,lv):
    x=tv.transforms.functional.adjust_brightness(x,lv["bright"]).clamp(0,1)
    return (x+lv["noise"]*torch.randn_like(x)).clamp(0,1)

class Tr(torch.utils.data.Dataset):
    def __init__(s,i): s.i=list(i)
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        x,y=full[s.i[k]]; return tf_tr(x),y

@torch.no_grad()
def probs_at(M,idxs,lv):
    ps,ys=[],[]
    for j in range(0,len(idxs),256):
        xb=[]; yb=[]
        for idx in idxs[j:j+256]:
            img,y=full[idx]
            x=degrade(tv.transforms.functional.to_tensor(img),lv)
            xb.append(norm(x)); yb.append(y)
        ps.append(torch.softmax(M(torch.stack(xb).to(dev)),1).cpu())
        ys.append(torch.tensor(yb))
    return torch.cat(ps),torch.cat(ys)

def qhat_of(p,y):
    s=1-p[torch.arange(len(y)),y]
    k=min(int((1-ALPHA)*(len(s)+1))/len(s),1.0)
    return torch.quantile(s,k).item()

rows=[]; t0=time.time()
for seed in SEEDS:
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed)
    i_tr,i_ca,i_te=random_split(range(n),[int(0.7*n),int(0.15*n),n-int(0.7*n)-int(0.15*n)],generator=g)
    i_ca,i_te=list(i_ca),list(i_te)
    M=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); M.fc=nn.Linear(512,10); M=M.to(dev)
    o=torch.optim.AdamW(M.parameters(),lr=3e-4,weight_decay=1e-4)
    sc=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=12)
    ltr=DataLoader(Tr(i_tr),batch_size=256,shuffle=True,num_workers=8,pin_memory=True)
    for e in range(12):
        M.train()
        for x,y in ltr:
            x,y=x.to(dev),y.to(dev); o.zero_grad(); F.cross_entropy(M(x),y).backward(); o.step()
        sc.step()
    M.eval()
    p0,y0=probs_at(M,i_ca[:3000],drift_level(0))
    qh_launch=qhat_of(p0,y0)
    qh_B=qh_launch; qh_C=qh_launch
    for week in range(WEEKS):
        lv=drift_level(week)
        te=i_te[(week*PER_WEEK)%len(i_te):][:PER_WEEK]
        p,y=probs_at(M,te,lv)
        s=1-p[torch.arange(len(y)),y]
        if week>0 and week%4==0:    # B refresh: 200 labeled samples this week
            st=(week*REFRESH_N)%(len(i_ca)-REFRESH_N)
            pr,yr=probs_at(M,i_ca[st:st+REFRESH_N],lv)
            qh_B=qhat_of(pr,yr)
        if week>0 and week%12==0:   # C refresh
            st=(1700+week*REFRESH_N)%(len(i_ca)-REFRESH_N)
            pr,yr=probs_at(M,i_ca[st:st+REFRESH_N],lv)
            qh_C=qhat_of(pr,yr)
        acc=(p.argmax(1)==y).float().mean().item()
        rows.append(dict(seed=seed,week=week,noise=lv["noise"],bright=lv["bright"],acc=acc,
            cov_A=(s<=qh_launch).float().mean().item(),
            cov_B=(s<=qh_B).float().mean().item(),
            cov_C=(s<=qh_C).float().mean().item(),
            set_B=float((p>=(1-qh_B)).sum(1).float().mean())))
    print(f"[s12] seed {seed} done ({time.time()-t0:.0f}s)",flush=True)

def ms(v): t=torch.tensor(v); return {"mean":float(t.mean()),"std":float(t.std()) if len(v)>1 else 0.0}
by_week={str(w):{k:ms([r[k] for r in rows if r["week"]==w])
    for k in ["acc","cov_A","cov_B","cov_C","set_B"]} for w in range(WEEKS)}
summary={"n_seeds":len(SEEDS),"alpha":ALPHA,"weeks":WEEKS,
    "refresh_samples":REFRESH_N,"refresh_bytes_estimate":REFRESH_N*4,
    "by_week":by_week,"runtime_s":round(time.time()-t0,1)}
with open(f"{OUT}/stage12_results.json","w") as f:
    json.dump({"summary":summary,"rows":rows},f,indent=2)
print(f"[s12] DONE ({summary['runtime_s']}s)",flush=True)
