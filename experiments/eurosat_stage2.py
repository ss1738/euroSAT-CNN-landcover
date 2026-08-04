#!/usr/bin/env python3
"""
Stage 2 — multi-seed sweep + OOD refusal behaviour.

For each of 20 seeds:
  train ResNet18 (12 epochs) -> FP32 + INT8 PTQ
  for alpha in {0.05, 0.10, 0.15}:
    FP32 coverage/set-size/empty-rate
    INT8 coverage with STALE fp32 qhat vs RECALIBRATED qhat
  OOD (alpha=0.10, INT8+recal): refusal (empty-set) rate + coverage under
    gaussian noise / blur / brightness corruptions vs clean.

Output: ~/eurosat-assured/sweep_results.json (per-run rows + aggregates)
"""
import json, os, time, copy
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS = list(range(20))
ALPHAS = [0.05, 0.10, 0.15]
EPOCHS = 12
BATCH = 256
ROOT = os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data")
OUT = os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured"))
dev = "cuda"

MEAN, STD = [0.3444, 0.3803, 0.4078], [0.0914, 0.0651, 0.0552]
norm = tv.transforms.Normalize(MEAN, STD)
tf_train = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip(),
    tv.transforms.ToTensor(), norm])
tf_eval = tv.transforms.Compose([tv.transforms.ToTensor(), norm])

def tf_corrupt(kind):
    def apply(img):
        x = tv.transforms.functional.to_tensor(img)
        if kind == "noise":
            x = (x + 0.10 * torch.randn_like(x)).clamp(0, 1)
        elif kind == "blur":
            x = tv.transforms.functional.gaussian_blur(x, 5, sigma=2.0)
        elif kind == "bright":
            x = tv.transforms.functional.adjust_brightness(x, 1.6).clamp(0, 1)
        return norm(x)
    return apply

full = tv.datasets.EuroSAT(root=ROOT, download=True)
n = len(full)
n_train, n_cal = int(0.70 * n), int(0.15 * n)
n_test = n - n_train - n_cal

class Sub(torch.utils.data.Dataset):
    def __init__(self, base, idxs, tf):
        self.base, self.idxs, self.tf = base, list(idxs), tf
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        img, y = self.base[self.idxs[i]]
        return self.tf(img), y

@torch.no_grad()
def softmaxes(m, loader, device):
    m.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.softmax(m(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)

def qhat_of(p_cal, y_cal, alpha):
    s = 1.0 - p_cal[torch.arange(len(y_cal)), y_cal]
    k = min(int((1 - alpha) * (len(s) + 1)) / len(s), 1.0)
    return torch.quantile(s, k).item()

def cp_eval(p, y, qhat):
    sets = p >= (1.0 - qhat)
    cover = sets[torch.arange(len(y)), y].float().mean().item()
    size = sets.sum(1).float().mean().item()
    empty = (sets.sum(1) == 0).float().mean().item()
    return cover, size, empty

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

rows, ood_rows = [], []
t_all = time.time()
for seed in SEEDS:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    i_tr, i_ca, i_te = random_split(range(n), [n_train, n_cal, n_test], generator=g)
    d_tr = Sub(full, i_tr, tf_train)
    l_tr = DataLoader(d_tr, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True)
    l_ca = DataLoader(Sub(full, i_ca, tf_eval), batch_size=BATCH, num_workers=4)
    l_te = DataLoader(Sub(full, i_te, tf_eval), batch_size=BATCH, num_workers=4)

    m = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 10)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        m.train()
        for x, y in l_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            lf(m(x), y).backward()
            opt.step()
        sch.step()

    p_ca, y_ca = softmaxes(m, l_ca, dev)
    p_te, y_te = softmaxes(m, l_te, dev)

    cm = copy.deepcopy(m).cpu().eval()
    prep = prepare_fx(cm, get_default_qconfig_mapping("fbgemm"),
                      example_inputs=(torch.randn(1, 3, 64, 64),))
    with torch.no_grad():
        seen = 0
        for x, _ in DataLoader(d_tr, batch_size=256, num_workers=4):
            prep(x)
            seen += len(x)
            if seen >= 2048:
                break
    q8 = convert_fx(prep)
    pq_ca, _ = softmaxes(q8, l_ca, "cpu")
    pq_te, _ = softmaxes(q8, l_te, "cpu")

    acc32 = (p_te.argmax(1) == y_te).float().mean().item()
    acc8 = (pq_te.argmax(1) == y_te).float().mean().item()

    for alpha in ALPHAS:
        qf = qhat_of(p_ca, y_ca, alpha)
        qi = qhat_of(pq_ca, y_ca, alpha)
        c32, s32, e32 = cp_eval(p_te, y_te, qf)
        cst, sst, est = cp_eval(pq_te, y_te, qf)   # stale
        crc, src, erc = cp_eval(pq_te, y_te, qi)   # recal
        rows.append(dict(seed=seed, alpha=alpha, acc_fp32=acc32, acc_int8=acc8,
                         cov_fp32=c32, cov_int8_stale=cst, cov_int8_recal=crc,
                         empty_fp32=e32, empty_int8_stale=est, empty_int8_recal=erc,
                         size_fp32=s32, size_int8_recal=src))

    # OOD at alpha=0.10 with recalibrated INT8 qhat
    qi10 = qhat_of(pq_ca, y_ca, 0.10)
    for kind in ["clean", "noise", "blur", "bright"]:
        tf_k = tf_eval if kind == "clean" else tf_corrupt(kind)
        l_k = DataLoader(Sub(full, i_te, tf_k), batch_size=BATCH, num_workers=4)
        pk, yk = softmaxes(q8, l_k, "cpu")
        ck, sk, ek = cp_eval(pk, yk, qi10)
        acck = (pk.argmax(1) == yk).float().mean().item()
        ood_rows.append(dict(seed=seed, corruption=kind, acc=acck,
                             coverage=ck, set_size=sk, refusal_rate=ek))
    print(f"[stage2] seed {seed} done ({time.time()-t_all:.0f}s elapsed)", flush=True)

def agg(rs, keys, group):
    out = {}
    for gval in sorted(set(r[group] for r in rs)):
        sel = [r for r in rs if r[group] == gval]
        out[str(gval)] = {k: {"mean": float(torch.tensor([r[k] for r in sel]).mean()),
                              "std": float(torch.tensor([r[k] for r in sel]).std())}
                          for k in keys}
    return out

summary = {
    "n_seeds": len(SEEDS),
    "by_alpha": agg(rows, ["cov_fp32", "cov_int8_stale", "cov_int8_recal",
                           "empty_fp32", "empty_int8_stale", "empty_int8_recal",
                           "acc_fp32", "acc_int8"], "alpha"),
    "ood_alpha0.10_int8recal": agg(ood_rows, ["acc", "coverage", "set_size",
                                              "refusal_rate"], "corruption"),
    "runtime_s": round(time.time() - t_all, 1),
}
with open(f"{OUT}/sweep_results.json", "w") as f:
    json.dump({"summary": summary, "rows": rows, "ood_rows": ood_rows}, f, indent=2)
print(f"[stage2] DONE -> {OUT}/sweep_results.json  ({summary['runtime_s']}s)", flush=True)
