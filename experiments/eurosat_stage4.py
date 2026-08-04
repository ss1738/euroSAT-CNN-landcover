#!/usr/bin/env python3
"""
Stage 4 — three-check gate + stronger MCU student.

Gate: ACCEPT iff (conformal set non-empty) AND (energy <= q95 cal)
               AND (input-stats inside cal envelope).
Input-stats monitor (raw tensor, pre-normalization, MCU-cheap):
  hf  = mean |x - avgpool3(x)|   (high-frequency energy)
  sd  = mean per-channel std
  envelope = [q0.5%, q99.5%] of each stat on calibration set; outside either -> OOD.

Student v2: width x3 (48/96/192), 50 KD epochs, ColorJitter aug.
5 seeds. Output: ~/eurosat-assured/stage4_results.json
"""
import json, os, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, random_split

SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.10
EPOCHS_T = 12
EPOCHS_S = 50
BATCH = 256
KD_T = 4.0
ROOT = os.path.join(os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured")), "data")
OUT = os.environ.get("EA_DIR", os.path.expanduser("~/eurosat-assured"))
dev = "cuda"

MEAN, STD = [0.3444, 0.3803, 0.4078], [0.0914, 0.0651, 0.0552]
norm = tv.transforms.Normalize(MEAN, STD)
tf_train = tv.transforms.Compose([
    tv.transforms.RandomHorizontalFlip(), tv.transforms.RandomVerticalFlip(),
    tv.transforms.ColorJitter(0.2, 0.2, 0.1),
    tv.transforms.ToTensor(), norm])

def raw_and_norm(kind):
    """Return transform producing (normalized_tensor, raw_tensor) pairs."""
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

def input_stats(xr):                      # xr: raw [B,3,H,W] in [0,1]
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

class TinyEOv2(nn.Module):
    def __init__(self, ncls=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 48, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(48), nn.ReLU())
        self.b1 = DSBlock(48, 96, 2)
        self.b2 = DSBlock(96, 192, 2)
        self.b3 = DSBlock(192, 192, 2)
        self.fc = nn.Linear(192, ncls)
    def forward(self, x):
        x = self.b3(self.b2(self.b1(self.stem(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

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

def energy(logits):
    return -torch.logsumexp(logits, dim=1)

def quantize(model, d_train_plain):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    cm = copy.deepcopy(model).cpu().eval()
    prep = prepare_fx(cm, get_default_qconfig_mapping("fbgemm"),
                      example_inputs=(torch.randn(1, 3, 64, 64),))
    with torch.no_grad():
        seen = 0
        for x, _ in DataLoader(d_train_plain, batch_size=256, num_workers=4):
            prep(x)
            seen += len(x)
            if seen >= 2048:
                break
    return convert_fx(prep)

rows = []
t0 = time.time()
for seed in SEEDS:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    i_tr, i_ca, i_te = random_split(range(n), [n_train, n_cal, n_test], generator=g)
    d_tr = SubTrain(full, i_tr, tf_train)
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

    S = TinyEOv2().to(dev)
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

    d_tr_plain = SubTrain(full, i_tr, tv.transforms.Compose([tv.transforms.ToTensor(), norm]))
    for name, model in [("teacher", T), ("student", S)]:
        q8 = quantize(model, d_tr_plain)
        lg_ca, hf_ca, sd_ca, y_ca = logits_stats(q8, l_ca, "cpu")
        p_ca = torch.softmax(lg_ca, 1)
        qh = qhat_of(p_ca, y_ca, ALPHA)
        e_thr = torch.quantile(energy(lg_ca), 0.95).item()
        env = {s: (torch.quantile(v, 0.005).item(), torch.quantile(v, 0.995).item())
               for s, v in [("hf", hf_ca), ("sd", sd_ca)]}
        for kind in ["clean", "noise", "blur", "bright"]:
            l_k = DataLoader(SubPair(full, i_te, raw_and_norm(kind)), batch_size=BATCH, num_workers=4)
            lg, hf, sd, yk = logits_stats(q8, l_k, "cpu")
            p = torch.softmax(lg, 1)
            nonempty = (p >= (1.0 - qh)).sum(1) > 0
            e_ok = energy(lg) <= e_thr
            in_ok = ((hf >= env["hf"][0]) & (hf <= env["hf"][1]) &
                     (sd >= env["sd"][0]) & (sd <= env["sd"][1]))
            accept = nonempty & e_ok & in_ok
            wrong = p.argmax(1) != yk
            rows.append(dict(
                seed=seed, model=name, corruption=kind,
                accept_rate=accept.float().mean().item(),
                refusal_conformal=(~nonempty).float().mean().item(),
                refusal_energy=(~e_ok).float().mean().item(),
                refusal_inputstats=(~in_ok).float().mean().item(),
                err_all=wrong.float().mean().item(),
                err_accepted=(wrong & accept).sum().item() / max(accept.sum().item(), 1),
            ))
        if seed == SEEDS[0]:
            torch.save(q8.state_dict(), f"{OUT}/{name}_int8_s4.pt")
    print(f"[stage4] seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

def agg(rs, keys):
    out = {}
    for mdl in ["teacher", "student"]:
        out[mdl] = {}
        for kind in ["clean", "noise", "blur", "bright"]:
            sel = [r for r in rs if r["model"] == mdl and r["corruption"] == kind]
            out[mdl][kind] = {k: {"mean": float(torch.tensor([r[k] for r in sel]).mean()),
                                  "std": float(torch.tensor([r[k] for r in sel]).std())}
                              for k in keys}
    return out

summary = {
    "n_seeds": len(SEEDS), "alpha": ALPHA,
    "gate": agg(rows, ["accept_rate", "refusal_conformal", "refusal_energy",
                       "refusal_inputstats", "err_all", "err_accepted"]),
    "student_params": sum(p.numel() for p in TinyEOv2().parameters()),
    "student_int8_bytes": os.path.getsize(f"{OUT}/student_int8_s4.pt"),
    "runtime_s": round(time.time() - t0, 1),
}
with open(f"{OUT}/stage4_results.json", "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2)
print(f"[stage4] DONE -> {OUT}/stage4_results.json ({summary['runtime_s']}s)", flush=True)
