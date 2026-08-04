#!/usr/bin/env python3
"""
Stage 6 — (a) compression ladder Pareto (width 1.0/0.5/0.25 students, KD, INT8,
full 3-check gate metrics), (b) SEU bit-flip injection study: does the gate
detect weight corruption?  Output: ~/eurosat-assured/stage6_results.json
"""
import json, os, time, copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.10
EPOCHS_T = 12
EPOCHS_S = 60
BATCH = 256
KD_T = 4.0
WIDTHS = [1.0, 0.5, 0.25]
SEU_KS = [1, 10, 100]
SEU_TRIALS = 10
ROOT = os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data")
OUT = os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured"))
dev = "cuda"

MEAN, STD = [0.3444, 0.3803, 0.4078], [0.0914, 0.0651, 0.0552]
norm = tv.transforms.Normalize(MEAN, STD)
tf_train = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip(),
    tv.transforms.ColorJitter(0.2, 0.2, 0.1), tv.transforms.ToTensor(), norm])
tf_plain = tv.transforms.Compose([tv.transforms.ToTensor(), norm])

def raw_and_norm(kind):
    def apply(img):
        x = tv.transforms.functional.to_tensor(img)
        if kind == "noise":
            x = (x + 0.10 * torch.randn_like(x)).clamp(0, 1)
        elif kind == "blur":
            x = tv.transforms.functional.gaussian_blur(x, 5, sigma=2.0)
        elif kind == "bright":
            x = tv.transforms.functional.adjust_brightness(x, 1.6).clamp(0, 1)
        return norm(x.clone()), x
    return apply

full = tv.datasets.EuroSAT(root=ROOT, download=True)
n = len(full)
n_train, n_cal = int(0.70 * n), int(0.15 * n)
n_test = n - n_train - n_cal

class SubPair(torch.utils.data.Dataset):
    def __init__(self, base, idxs, tf):
        self.base, self.idxs, self.tf = base, list(idxs), tf
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        img, y = self.base[self.idxs[i]]
        xn, xr = self.tf(img)
        return xn, xr, y

class SubTrain(torch.utils.data.Dataset):
    def __init__(self, base, idxs, tf):
        self.base, self.idxs, self.tf = base, list(idxs), tf
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        img, y = self.base[self.idxs[i]]
        return self.tf(img), y

def input_stats(xr):
    hf = (xr - F.avg_pool2d(xr, 3, 1, 1)).abs().mean(dim=(1, 2, 3))
    sd = xr.std(dim=(2, 3)).mean(dim=1)
    return hf, sd

class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        return F.relu(self.bn2(self.pw(x)))

class TinyEOw(nn.Module):
    def __init__(self, w=1.0, ncls=10):
        super().__init__()
        c = [max(8, int(48 * w)), max(16, int(96 * w)), max(32, int(192 * w))]
        self.stem = nn.Sequential(nn.Conv2d(3, c[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(c[0]), nn.ReLU())
        self.b1 = DSBlock(c[0], c[1], 2)
        self.b2 = DSBlock(c[1], c[2], 2)
        self.b3 = DSBlock(c[2], c[2], 2)
        self.fc = nn.Linear(c[2], ncls)
    def forward(self, x):
        x = self.b3(self.b2(self.b1(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

@torch.no_grad()
def logits_stats(m, loader, device):
    m.eval()
    ls, hfs, sds, ys = [], [], [], []
    for xn, xr, y in loader:
        ls.append(m(xn.to(device)).cpu())
        hf, sd = input_stats(xr)
        hfs.append(hf)
        sds.append(sd)
        ys.append(y)
    return torch.cat(ls), torch.cat(hfs), torch.cat(sds), torch.cat(ys)

def qhat_of(p_cal, y_cal, alpha):
    s = 1.0 - p_cal[torch.arange(len(y_cal)), y_cal]
    k = min(int((1 - alpha) * (len(s) + 1)) / len(s), 1.0)
    return torch.quantile(s, k).item()

def energy(lg):
    return -torch.logsumexp(lg, dim=1)

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def ptq(model, d_plain):
    cm = copy.deepcopy(model).cpu().eval()
    prep = prepare_fx(cm, get_default_qconfig_mapping("fbgemm"),
                      example_inputs=(torch.randn(1, 3, 64, 64),))
    with torch.no_grad():
        seen = 0
        for x, _ in DataLoader(d_plain, batch_size=256, num_workers=4):
            prep(x)
            seen += len(x)
            if seen >= 2048:
                break
    return convert_fx(prep)

def gate_thresholds(model, l_ca, device):
    lg_ca, hf_ca, sd_ca, y_ca = logits_stats(model, l_ca, device)
    p_ca = torch.softmax(lg_ca, 1)
    return dict(qh=qhat_of(p_ca, y_ca, ALPHA),
                e_thr=torch.quantile(energy(lg_ca), 0.95).item(),
                hf_lo=torch.quantile(hf_ca, 0.005).item(),
                hf_hi=torch.quantile(hf_ca, 0.995).item(),
                sd_lo=torch.quantile(sd_ca, 0.005).item(),
                sd_hi=torch.quantile(sd_ca, 0.995).item())

def gate_eval(model, thr, loader, device):
    lg, hf, sd, yk = logits_stats(model, loader, device)
    p = torch.softmax(lg, 1)
    nonempty = (p >= (1.0 - thr["qh"])).sum(1) > 0
    e = energy(lg)
    e_ok = torch.isfinite(e) & (e <= thr["e_thr"])
    in_ok = ((hf >= thr["hf_lo"]) & (hf <= thr["hf_hi"]) &
             (sd >= thr["sd_lo"]) & (sd <= thr["sd_hi"]))
    accept = nonempty & e_ok & in_ok
    wrong = p.argmax(1) != yk
    return dict(acc=(~wrong).float().mean().item(),
                accept_rate=accept.float().mean().item(),
                err_accepted=(wrong & accept).sum().item() / max(accept.sum().item(), 1))

def bitflip(model, k, rng):
    m = copy.deepcopy(model)
    params = [p for p in m.parameters() if p.dtype == torch.float32 and p.numel() > 0]
    sizes = torch.tensor([p.numel() for p in params])
    total = int(sizes.sum())
    for _ in range(k):
        gi = rng.randrange(total)
        for p, s in zip(params, sizes):
            if gi < int(s):
                flat = p.data.view(-1)
                iv = flat[gi].view(torch.int32) if hasattr(flat[gi], 'view') else None
                bit = rng.randrange(32)
                v = flat[gi:gi+1].view(torch.int32)
                v ^= (1 << bit)
                break
            gi -= int(s)
    return m

rows, seu_rows = [], []
t0 = time.time()
for seed in SEEDS:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    i_tr, i_ca, i_te = random_split(range(n), [n_train, n_cal, n_test], generator=g)
    d_tr = SubTrain(full, i_tr, tf_train)
    d_plain = SubTrain(full, i_tr, tf_plain)
    l_tr = DataLoader(d_tr, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True)
    l_ca = DataLoader(SubPair(full, i_ca, raw_and_norm("clean")), batch_size=BATCH, num_workers=4)

    T = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    T.fc = nn.Linear(512, 10)
    T = T.to(dev)
    opt = torch.optim.AdamW(T.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_T)
    for ep in range(EPOCHS_T):
        T.train()
        for x, y in l_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            F.cross_entropy(T(x), y).backward()
            opt.step()
        sch.step()

    students = {}
    for w in WIDTHS:
        S = TinyEOw(w).to(dev)
        opt = torch.optim.AdamW(S.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_S)
        T.eval()
        for ep in range(EPOCHS_S):
            S.train()
            for x, y in l_tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.no_grad():
                    tl = T(x)
                sl = S(x)
                kd = F.kl_div(F.log_softmax(sl / KD_T, 1), F.softmax(tl / KD_T, 1),
                              reduction="batchmean") * KD_T * KD_T
                loss = 0.7 * kd + 0.3 * F.cross_entropy(sl, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
        students[w] = S

    # ladder metrics (INT8)
    for w, S in students.items():
        q8 = ptq(S, d_plain)
        thr = gate_thresholds(q8, l_ca, "cpu")
        nb = sum(p.numel() for p in S.parameters())
        torch.save(q8.state_dict(), f"{OUT}/ladder_w{w}_s{seed}.pt")
        size_b = os.path.getsize(f"{OUT}/ladder_w{w}_s{seed}.pt")
        for kind in ["clean", "noise", "blur", "bright"]:
            l_k = DataLoader(SubPair(full, i_te, raw_and_norm(kind)), batch_size=BATCH, num_workers=4)
            m = gate_eval(q8, thr, l_k, "cpu")
            m.update(seed=seed, width=w, params=nb, int8_bytes=size_b, corruption=kind)
            rows.append(m)

    # SEU study (seed 0 only): FP32 weight bit-flips, gate thresholds from clean cal
    if seed == 0:
        l_te_clean = DataLoader(SubPair(full, i_te, raw_and_norm("clean")), batch_size=BATCH, num_workers=4)
        for mname, mm in [("teacher", T.cpu()), ("student_w1.0", students[1.0].cpu())]:
            thr = gate_thresholds(mm, l_ca, "cpu")
            base = gate_eval(mm, thr, l_te_clean, "cpu")
            seu_rows.append(dict(model=mname, k=0, trial=0, **base))
            for k in SEU_KS:
                for tr in range(SEU_TRIALS):
                    rng = random.Random(1000 * k + tr)
                    mf = bitflip(mm, k, rng)
                    m = gate_eval(mf, thr, l_te_clean, "cpu")
                    seu_rows.append(dict(model=mname, k=k, trial=tr, **m))
            print(f"[stage6] SEU {mname} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[stage6] seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

def mstd(vals):
    t = torch.tensor(vals)
    return {"mean": float(t.mean()), "std": float(t.std()) if len(vals) > 1 else 0.0}

ladder = {}
for w in WIDTHS:
    ladder[str(w)] = {"params": [r["params"] for r in rows if r["width"] == w][0],
                      "int8_bytes": [r["int8_bytes"] for r in rows if r["width"] == w][0]}
    for kind in ["clean", "noise", "blur", "bright"]:
        sel = [r for r in rows if r["width"] == w and r["corruption"] == kind]
        ladder[str(w)][kind] = {k: mstd([r[k] for r in sel])
                                for k in ["acc", "accept_rate", "err_accepted"]}

seu = {}
for mname in ["teacher", "student_w1.0"]:
    seu[mname] = {}
    for k in [0] + SEU_KS:
        sel = [r for r in seu_rows if r["model"] == mname and r["k"] == k]
        seu[mname][str(k)] = {kk: mstd([r[kk] for r in sel])
                              for kk in ["acc", "accept_rate", "err_accepted"]}

summary = {"n_seeds": len(SEEDS), "alpha": ALPHA, "ladder": ladder,
           "seu_fp32_bitflips": seu, "seu_trials": SEU_TRIALS,
           "runtime_s": round(time.time() - t0, 1)}
with open(f"{OUT}/stage6_results.json", "w") as f:
    json.dump({"summary": summary, "rows": rows, "seu_rows": seu_rows}, f, indent=2)
print(f"[stage6] DONE -> {OUT}/stage6_results.json ({summary['runtime_s']}s)", flush=True)
