"""
ANP (Adversarial Neuron Pruning) — NeurIPS 2021
Official repo: https://github.com/csdongxian/ANP_backdoor

Core algorithm functions taken verbatim from the official implementation:
  - include_noise / exclude_noise / reset / clip_mask / sign_grad
    from optimize_mask_cifar.py (global args/device → explicit params)
  - mask_train()       verbatim from optimize_mask_cifar.py
  - pruning()          verbatim from prune_neuron_cifar.py
  - evaluate_by_threshold()  verbatim from prune_neuron_cifar.py
    (returns dicts instead of text lines; clean-only for non-oracle selection)
  - test()             verbatim from prune_neuron_cifar.py
  - save_mask_scores() verbatim from optimize_mask_cifar.py
  - load_state_dict()  verbatim from optimize_mask_cifar.py
    (extended: also handles our checkpoint format with "model" key)

Necessary (minimal) adaptations for pretrained checkpoints:
  - _replace_bn(): swaps BatchNorm2d → NoisyBatchNorm2d in-place so that
    exp1 checkpoints (trained with standard BN) can be used.

ANP class: thin wrapper that wires the official functions into our experiment
pipeline. Key differences from buggy version fixed here:
  - Threshold sweep runs on a copy; final pruning applied on a fresh copy at
    best_threshold only (fixes cumulative-state bug).
  - Threshold selection uses clean BA drop ≤ tol (non-oracle).
  - No fine-tuning step (not part of official ANP paper pipeline).
"""

import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

from .anp_batchnorm import NoisyBatchNorm2d, NoisyBatchNorm1d


# ─────────────────────────────────────────────────────────────────────────────
# Official helper functions — verbatim from optimize_mask_cifar.py
# (only change: `models.NoisyBatchNorm*` → local import; device/args → params)
# ─────────────────────────────────────────────────────────────────────────────

def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init, eps):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=eps)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader,
               device, anp_eps, anp_steps, anp_alpha):
    """Verbatim from optimize_mask_cifar.py mask_train(); global→param."""
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if anp_eps > 0.0:
            reset(model, rand_init=True, eps=anp_eps)
            for _ in range(anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = anp_alpha * loss_nat + (1 - anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def save_mask_scores(state_dict, file_name):
    """Verbatim from optimize_mask_cifar.py."""
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(
                    count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


def load_state_dict(net, orig_state_dict):
    """
    Verbatim from optimize_mask_cifar.py load_state_dict().
    Extended: also handles our checkpoint format with 'model' key.
    """
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if 'model' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['model']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Official pruning functions — verbatim from prune_neuron_cifar.py
# (device passed explicitly to test(); results returned as dicts, not text)
# ─────────────────────────────────────────────────────────────────────────────

def pruning(net, neuron):
    """Verbatim from prune_neuron_cifar.py pruning()."""
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_threshold(model, mask_values, pruning_max, pruning_step,
                           criterion, clean_loader, device):
    """
    Adapted from prune_neuron_cifar.py evaluate_by_threshold().
    Changes from original:
      - device passed as parameter (was global)
      - returns list of dicts instead of text lines
      - evaluates clean accuracy only (caller handles ASR with target_label)
    """
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        cl_loss, cl_acc = test(model=model, criterion=criterion,
                               data_loader=clean_loader, device=device)
        results.append({
            'threshold': round(float(threshold), 4),
            'n_pruned':  start,
            'cl_loss':   round(cl_loss, 4),
            'cl_acc':    round(cl_acc * 100.0, 2),  # fraction → percentage
        })
    return results


def test(model, criterion, data_loader, device):
    """Verbatim from prune_neuron_cifar.py test(); device passed as parameter."""
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


# ─────────────────────────────────────────────────────────────────────────────
# Necessary adaptation: BatchNorm2d → NoisyBatchNorm2d replacement
# (required because exp1 checkpoints were trained with standard BN;
#  the official repo builds models with NoisyBN from scratch via norm_layer=)
# ─────────────────────────────────────────────────────────────────────────────

def _replace_bn(module: nn.Module) -> nn.Module:
    """Replace all BatchNorm2d with NoisyBatchNorm2d in-place (recursive)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            noisy = NoisyBatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            if child.affine:
                with torch.no_grad():
                    noisy.weight.data.copy_(child.weight.data)
                    noisy.bias.data.copy_(child.bias.data)
            if child.track_running_stats:
                noisy.running_mean.copy_(child.running_mean)
                noisy.running_var.copy_(child.running_var)
                noisy.num_batches_tracked.copy_(child.num_batches_tracked)
            setattr(module, name, noisy)
        else:
            _replace_bn(child)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# ANP class — thin experiment wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ANP:
    """
    Wires official ANP functions into our experiment pipeline.

    Pipeline (follows official paper):
      1. Replace BN → NoisyBN  (via _replace_bn; official uses norm_layer=)
      2. Mask optimization     (official mask_train, 2000 iter, 1% clean data)
      3. Threshold sweep       (official evaluate_by_threshold on a model copy)
      4. Select best threshold (clean BA drop ≤ ba_drop_tol — non-oracle)
      5. Apply pruning         (official pruning() on a fresh copy at best_thr)
      6. Evaluate final BA/ASR

    No fine-tuning: not part of the official ANP pipeline.
    """

    def __init__(
        self,
        model:        nn.Module,
        device:       str   = "cuda",
        anp_eps:      float = 0.4,
        anp_steps:    int   = 1,
        anp_alpha:    float = 0.2,
        lr:           float = 0.2,
        nb_iter:      int   = 2000,
        pruning_step: float = 0.05,
        pruning_max:  float = 0.95,
        ba_drop_tol:  float = 2.0,
    ):
        self.device       = device
        self.anp_eps      = anp_eps
        self.anp_steps    = anp_steps
        self.anp_alpha    = anp_alpha
        self.lr           = lr
        self.nb_iter      = nb_iter
        self.pruning_step = pruning_step
        self.pruning_max  = pruning_max
        self.ba_drop_tol  = ba_drop_tol
        self._model_src   = model   # original model untouched

    def run(
        self,
        clean_loader:        DataLoader,  # 1% clean train data (mask opt)
        eval_clean_loader:   DataLoader,  # clean test set, target excluded (BA)
        eval_poison_loader:  DataLoader,  # poison test set (ASR)
        target_label:        int,
        original_ba:         float,
        eval_full_loader:    DataLoader = None,  # all-class clean test (BA_full)
        ft_epochs:           int   = 0,   # ignored (not in official ANP pipeline)
        ft_lr:               float = 0.001,
    ) -> dict:
        from utils.metrics import compute_asr

        device    = self.device
        criterion = nn.CrossEntropyLoss().to(device)

        # ── Step 1: build NoisyBN model from original ─────────────────────────
        print("[ANP] Step 1: BatchNorm2d → NoisyBatchNorm2d")
        model = _replace_bn(copy.deepcopy(self._model_src)).to(device)
        load_state_dict(model, self._model_src.state_dict())

        # ── Step 2: optimizers (verbatim from optimize_mask_cifar.py main()) ──
        parameters   = list(model.named_parameters())
        mask_params  = [v for n, v in parameters if "neuron_mask"  in n]
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        mask_opt  = torch.optim.SGD(mask_params,  lr=self.lr, momentum=0.9)
        noise_opt = torch.optim.SGD(noise_params,
                                    lr=self.anp_eps / max(self.anp_steps, 1))

        # ── Step 3: mask optimization ─────────────────────────────────────────
        # Official: nb_repeat calls to mask_train, each over print_every batches
        # via RandomSampler(num_samples=print_every * batch_size)
        PRINT_EVERY = 500
        nb_repeat   = int(np.ceil(self.nb_iter / PRINT_EVERY))
        batch_size  = clean_loader.batch_size or 128
        ds          = clean_loader.dataset

        print(f"[ANP] Step 2: mask optimization "
              f"({nb_repeat}×{PRINT_EVERY}={nb_repeat*PRINT_EVERY} iter, "
              f"eps={self.anp_eps}, alpha={self.anp_alpha})")

        for rep in range(nb_repeat):
            sampler    = RandomSampler(ds, replacement=True,
                                       num_samples=PRINT_EVERY * batch_size)
            rep_loader = DataLoader(ds, batch_size=batch_size,
                                    sampler=sampler, num_workers=0)
            tr_loss, tr_acc = mask_train(
                model, criterion, mask_opt, noise_opt, rep_loader,
                device, self.anp_eps, self.anp_steps, self.anp_alpha,
            )
            print(f"  [{(rep+1)*PRINT_EVERY:>5}/{nb_repeat*PRINT_EVERY}] "
                  f"loss={tr_loss:.4f}  acc={tr_acc:.4f}")

        exclude_noise(model)  # ensure clean mode after training

        # ── Step 4: extract mask scores ───────────────────────────────────────
        mask_values = []
        for name, param in model.state_dict().items():
            if 'neuron_mask' in name:
                layer = '.'.join(name.split('.')[:-1])
                for idx in range(param.size(0)):
                    mask_values.append((layer, idx, param[idx].item()))
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        print(f"[ANP] {len(mask_values)} neuron mask scores extracted")

        # ── Step 5: threshold sweep (on a copy — cumulative in-place pruning) ─
        print("[ANP] Step 3: threshold sweep (clean BA only, non-oracle)")
        model_sweep  = copy.deepcopy(model)
        sweep_results = evaluate_by_threshold(
            model_sweep, mask_values, self.pruning_max, self.pruning_step,
            criterion, eval_clean_loader, device,
        )
        for r in sweep_results:
            print(f"  thr={r['threshold']:.2f}  pruned={r['n_pruned']:4d}  "
                  f"BA={r['cl_acc']:.2f}%")

        # ── Step 6: select best threshold (non-oracle) ────────────────────────
        best_row = None
        for r in sweep_results:
            if r['cl_acc'] >= original_ba - self.ba_drop_tol:
                best_row = r   # keep iterating → last valid = highest threshold
        if best_row is None:
            # BA never stays in tolerance → pick row with highest BA
            best_row = max(sweep_results, key=lambda r: r['cl_acc'])
        best_threshold = best_row['threshold']
        n_pruned       = best_row['n_pruned']
        print(f"[ANP] best_threshold={best_threshold}  "
              f"n_pruned={n_pruned}  BA@thr={best_row['cl_acc']:.2f}%")

        # ── Step 7: apply pruning at best_threshold on fresh copy ─────────────
        # (fixes bug: sweep runs to pruning_max; we must start fresh at thr only)
        model_final = copy.deepcopy(model)
        for neuron in mask_values:
            if float(neuron[2]) <= best_threshold:
                pruning(model_final, neuron)
            else:
                break

        # ── Step 8: evaluate ──────────────────────────────────────────────────
        model_final.eval()
        _, ba_frac = test(model_final, criterion, eval_clean_loader, device)
        ba_final   = round(ba_frac * 100.0, 2)
        asr_final  = compute_asr(model_final, eval_poison_loader,
                                 target_label, device)
        bypass = bool(asr_final > 50.0)

        # BA_full: accuracy on all classes (standard ANP metric, no target exclusion)
        ba_full = None
        if eval_full_loader is not None:
            _, ba_full_frac = test(model_final, criterion, eval_full_loader, device)
            ba_full = round(ba_full_frac * 100.0, 2)

        print(f"[ANP] Final: BA={ba_final:.2f}%"
              f"{f'  BA_full={ba_full:.2f}%' if ba_full is not None else ''}"
              f"  ASR={asr_final:.2f}%  "
              f"bypass={'YES (attack survives)' if bypass else 'NO (defense wins)'}")

        return {
            "BA":             ba_final,
            "BA_full":        ba_full,
            "ASR":            asr_final,
            "bypass":         bypass,
            "best_threshold": best_threshold,
            "n_pruned":       n_pruned,
            "sweep":          sweep_results,
        }
