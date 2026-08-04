#!/usr/bin/env python3
"""
Stage 3 — (A) energy-score OOD detection to catch the noise blind spot,
          (B) MCU-class distilled student + INT8, full assured-gate metrics.

Gate rule evaluated:  ACCEPT iff conformal set non-empty AND energy <= threshold
(threshold = 95th percentile of calibration energies -> ~5% clean FPR budget).

Per condition (clean/noise/blur/bright) x model (teacher-INT8, student-INT8):
  accept_rate, refusal_conformal, refusal_ood, refusal_either,
  err_all (top-1 error on everything), err_accepted (error among ACCEPTED = residual risk)

5 seeds. Output: ~/eurosat-assured/stage3_results.json
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
EPOCHS_S = 25
BATCH = 256
KD_T = 4.0
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

class TinyEO(nn.Module):
    """MCU-class depthwise-separable CNN for 64x64 EO tiles."""
    def __init__(self, ncls=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(32), nn.ReLU())
        self.b1 = DSBlock(32, 64, 2)
        self.b2 = DSBlock(64, 128, 2)
        self.b3 = DSBlock(128, 128, 2)
        self.fc = nn.Linear(128, ncls)
    def forward(self, x):
        x = self.b3(self.b2(self.b1(self.stem(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

@torch.no_grad()
def logits_of(m, loader, device):
    m.eval()
    ls, ys = [], []
    for x, y in loader:
        ls.append(m(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(ls), torch.cat(ys)

def qhat_of(p_cal, y_cal, alpha):
    s = 1.0 - p_cal[torch.arange(len(y_cal)), y_cal]
    k = min(int((1 - alpha) * (len(s) + 1)) / len(s), 1.0)
    return torch.quantile(s, k).item()

def energy(logits):                      # higher = more OOD
    return -torch.logsumexp(logits, dim=1)

def quantize(model, d_train):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    cm = copy.deepcopy(model).cpu().eval()
    prep = prepare_fx(cm, get_default_qconfig_mapping("fbgemm"),
                      example_inputs=(torch.randn(1, 3, 64, 64),))
    with torch.no_grad():
        seen = 0
        for x, _ in DataLoader(d_train, batch_size=256, num_workers=4):
            prep(x)
            seen += len(x)
            if seen >= 2048:
                break
    return convert_fx(prep)

def gate_metrics(logits, y, qhat, e_thr):
    p = torch.softmax(logits, dim=1)
    sets = p >= (1.0 - qhat)
    nonempty = sets.sum(1) > 0
    e = energy(logits)
    in_dist = e <= e_thr
    accept = nonempty & in_dist
    wrong = p.argmax(1) != y
    m = dict(
        accept_rate=accept.float().mean().item(),
        refusal_conformal=(~nonempty).float().mean().item(),
        refusal_ood=(~in_dist).float().mean().item(),
        refusal_either=(~accept).float().mean().item(),
        err_all=wrong.float().mean().item(),
        err_accepted=(wrong & accept).sum().item() / max(accept.sum().item(), 1),
    )
    return m

rows = []
t0 = time.time()
for seed in SEEDS:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    i_tr, i_ca, i_te = random_split(range(n), [n_train, n_cal, n_test], generator=g)
    d_tr = Sub(full, i_tr, tf_train)
    l_tr = DataLoader(d_tr, batch_size=BATCH, shuffle=True, num_workers=8, pin_memory=True)
    l_ca = DataLoader(Sub(full, i_ca, tf_eval), batch_size=BATCH, num_workers=4)

    # teacher
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

    # student via KD
    S = TinyEO().to(dev)
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

    for name, model in [("teacher", T), ("student", S)]:
        q8 = quantize(model, d_tr)
        lg_ca, y_ca = logits_of(q8, l_ca, "cpu")
        p_ca = torch.softmax(lg_ca, 1)
        qh = qhat_of(p_ca, y_ca, ALPHA)
        e_thr = torch.quantile(energy(lg_ca), 0.95).item()
        for kind in ["clean", "noise", "blur", "bright"]:
            tf_k = tf_eval if kind == "clean" else tf_corrupt(kind)
            l_k = DataLoader(Sub(full, i_te, tf_k), batch_size=BATCH, num_workers=4)
            lg, yk = logits_of(q8, l_k, "cpu")
            m = gate_metrics(lg, yk, qh, e_thr)
            m.update(seed=seed, model=name, corruption=kind)
            rows.append(m)
        if seed == SEEDS[0]:
            torch.save(q8.state_dict(), f"{OUT}/{name}_int8_s3.pt")
    nparam_s = sum(p.numel() for p in S.parameters())
    print(f"[stage3] seed {seed} done ({time.time()-t0:.0f}s) student_params={nparam_s}", flush=True)

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
    "n_seeds": len(SEEDS), "alpha": ALPHA, "ood_rule": "energy > q95(cal)",
    "gate": agg(rows, ["accept_rate", "refusal_conformal", "refusal_ood",
                       "refusal_either", "err_all", "err_accepted"]),
    "student_params": sum(p.numel() for p in TinyEO().parameters()),
    "student_int8_bytes": os.path.getsize(f"{OUT}/student_int8_s3.pt"),
    "teacher_int8_bytes": os.path.getsize(f"{OUT}/teacher_int8_s3.pt"),
    "runtime_s": round(time.time() - t0, 1),
}
with open(f"{OUT}/stage3_results.json", "w") as f:
    json.dump({"summary": summary, "rows": rows}, f, indent=2)
print(f"[stage3] DONE -> {OUT}/stage3_results.json ({summary['runtime_s']}s)", flush=True)
