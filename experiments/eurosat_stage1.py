#!/usr/bin/env python3
"""
Stage 1 — Assured onboard EO stack: baseline + conformal + INT8 coverage delta.

Measures (all written to results.json):
  1. FP32 ResNet18 EuroSAT accuracy (train/cal/test split 70/15/15, seed 42)
  2. Split-conformal (LAC score, alpha=0.10): qhat on cal, empirical coverage +
     mean prediction-set size on test
  3. INT8 post-training static quantization (FX graph mode, CPU):
     a) accuracy delta vs FP32
     b) coverage with STALE FP32 qhat reused on INT8 model  <- does quantization break coverage?
     c) coverage after recalibrating qhat on INT8 outputs   <- is recalibration sufficient?
  4. Model sizes (FP32 vs INT8 state_dict bytes)

Run: nohup python3 eurosat_stage1.py > stage1.log 2>&1 &
"""
import json, os, time, copy
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEED = 42
ALPHA = 0.10
EPOCHS = 12
BATCH = 256
ROOT = os.path.expanduser("~/eurosat-assured/data")
OUT = os.path.expanduser("~/eurosat-assured")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[stage1] device={dev}", flush=True)

# ---------- data ----------
tf_train = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomVerticalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0914, 0.0651, 0.0552]),
])
tf_eval = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.3444, 0.3803, 0.4078], [0.0914, 0.0651, 0.0552]),
])

full = tv.datasets.EuroSAT(root=ROOT, download=True)
n = len(full)
n_train, n_cal = int(0.70 * n), int(0.15 * n)
n_test = n - n_train - n_cal
g = torch.Generator().manual_seed(SEED)
idx_train, idx_cal, idx_test = random_split(range(n), [n_train, n_cal, n_test], generator=g)

class Sub(torch.utils.data.Dataset):
    def __init__(self, base, idxs, tf):
        self.base, self.idxs, self.tf = base, list(idxs), tf
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        img, y = self.base[self.idxs[i]]
        return self.tf(img), y

d_train = Sub(full, idx_train, tf_train)
d_cal = Sub(full, idx_cal, tf_eval)
d_test = Sub(full, idx_test, tf_eval)
print(f"[stage1] n={n} train={len(d_train)} cal={len(d_cal)} test={len(d_test)}", flush=True)

l_train = DataLoader(d_train, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True)
l_cal = DataLoader(d_cal, batch_size=BATCH, num_workers=4)
l_test = DataLoader(d_test, batch_size=BATCH, num_workers=4)

# ---------- train FP32 ----------
model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 10)
model = model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
lossf = nn.CrossEntropyLoss()

t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    tot = cor = 0
    for x, y in l_train:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
        tot += y.numel()
        cor += (out.argmax(1) == y).sum().item()
    sched.step()
    print(f"[stage1] epoch {ep+1}/{EPOCHS} train_acc={cor/tot:.4f}", flush=True)
train_time = time.time() - t0

@torch.no_grad()
def softmaxes(m, loader, device):
    m.eval()
    ps, ys = [], []
    for x, y in loader:
        p = torch.softmax(m(x.to(device)), dim=1).cpu()
        ps.append(p)
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)

def accuracy(p, y):
    return (p.argmax(1) == y).float().mean().item()

def conformal_qhat(p_cal, y_cal, alpha):
    scores = 1.0 - p_cal[torch.arange(len(y_cal)), y_cal]   # LAC nonconformity
    k = min(int((1 - alpha) * (len(scores) + 1)) / len(scores), 1.0)
    return torch.quantile(scores, k).item()

def conformal_eval(p, y, qhat):
    sets = p >= (1.0 - qhat)                                 # class in set iff softmax >= 1-qhat
    cover = sets[torch.arange(len(y)), y].float().mean().item()
    size = sets.sum(1).float().mean().item()
    return cover, size

p_cal, y_cal = softmaxes(model, l_cal, dev)
p_test, y_test = softmaxes(model, l_test, dev)
acc_fp32 = accuracy(p_test, y_test)
qhat_fp32 = conformal_qhat(p_cal, y_cal, ALPHA)
cov_fp32, size_fp32 = conformal_eval(p_test, y_test, qhat_fp32)
print(f"[stage1] FP32 acc={acc_fp32:.4f} qhat={qhat_fp32:.4f} coverage={cov_fp32:.4f} set_size={size_fp32:.3f}", flush=True)

torch.save(model.state_dict(), f"{OUT}/resnet18_fp32.pt")

# ---------- INT8 PTQ (FX graph mode, CPU) ----------
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

cpu_model = copy.deepcopy(model).cpu().eval()
qmap = get_default_qconfig_mapping("fbgemm")
example = torch.randn(1, 3, 64, 64)
prepared = prepare_fx(cpu_model, qmap, example_inputs=(example,))
with torch.no_grad():                       # calibrate observers on training data (CPU)
    seen = 0
    for x, _ in DataLoader(d_train, batch_size=256, num_workers=4):
        prepared(x)
        seen += len(x)
        if seen >= 2048:
            break
int8 = convert_fx(prepared)

p_cal_q, _ = softmaxes(int8, l_cal, "cpu")
p_test_q, _ = softmaxes(int8, l_test, "cpu")
acc_int8 = accuracy(p_test_q, y_test)
cov_stale, size_stale = conformal_eval(p_test_q, y_test, qhat_fp32)      # stale FP32 qhat
qhat_int8 = conformal_qhat(p_cal_q, y_cal, ALPHA)                        # recalibrated
cov_recal, size_recal = conformal_eval(p_test_q, y_test, qhat_int8)
print(f"[stage1] INT8 acc={acc_int8:.4f}", flush=True)
print(f"[stage1] INT8+stale-qhat  coverage={cov_stale:.4f} set_size={size_stale:.3f}", flush=True)
print(f"[stage1] INT8+recal-qhat  coverage={cov_recal:.4f} set_size={size_recal:.3f}", flush=True)

torch.save(int8.state_dict(), f"{OUT}/resnet18_int8.pt")
sz_fp32 = os.path.getsize(f"{OUT}/resnet18_fp32.pt")
sz_int8 = os.path.getsize(f"{OUT}/resnet18_int8.pt")

results = {
    "seed": SEED, "alpha": ALPHA, "epochs": EPOCHS, "train_time_s": round(train_time, 1),
    "split": {"train": len(d_train), "cal": len(d_cal), "test": len(d_test)},
    "fp32": {"acc": acc_fp32, "qhat": qhat_fp32, "coverage": cov_fp32, "set_size": size_fp32,
             "size_bytes": sz_fp32},
    "int8": {"acc": acc_int8, "size_bytes": sz_int8,
             "stale_qhat": {"coverage": cov_stale, "set_size": size_stale},
             "recal_qhat": {"qhat": qhat_int8, "coverage": cov_recal, "set_size": size_recal}},
    "target_coverage": 1 - ALPHA,
}
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[stage1] DONE -> {OUT}/results.json", flush=True)
