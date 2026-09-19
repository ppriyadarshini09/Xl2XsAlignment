# @title Posthoc Alignment

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_gpt import get_batch, device 


def _fixed_perm(n, device, seed=4242):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g).to(device)

@torch.no_grad()
def _geom_one_batch(xs_model, xs_scale, xl_model, xl_scale, site, head, x, y):
  """All alignment geometry for a single eval batch."""
  V = xs_model.lm_head.out_features
  D = xs_model.n_embed
  NAN = float('nan')

  xs = xs_model.forward_repr_at_site(site, x)
  xs = xs / xs_scale

  xl = xl_model.forward_repr_at_site(site, x)
  xl = xl/ xl_scale

  xl2xs = head(xl)

  def ce(repr_):
    logits = xs_model.lm_head(repr_) if repr_.shape[-1] == D else xl_model.lm_head(repr_)
    B, T, _ = logits.shape
    return F.cross_entropy(logits.view(B*T, V), y.view(B*T)).item()

  flat = lambda t: t.reshape(-1, t.shape[-1])
  xs_f, xl_f, xl2xs_f = flat(xs), flat(xl), flat(xl2xs)

  matched = F.cosine_similarity(xs_f, xl2xs_f, dim=-1)
  matched_sim = matched.mean().item()
  perm = _fixed_perm(xs_f.shape[0], xs_f.device)
  shuffled_sim = F.cosine_similarity(xs_f[perm], xl2xs_f, dim=-1).mean().item()
  centered_sim = F.cosine_similarity(
      xs_f - xs_f.mean(0, keepdim=True),
      xl2xs_f - xl2xs_f.mean(0, keepdim=True), dim=-1).mean().item()

  xs_norm = xs_f.norm(dim=-1).mean().item()
  xl_norm = xl_f.norm(dim=-1).mean().item()
  xl2xs_norm = xl2xs_f.norm(dim=-1).mean().item()

  # bidirectional-only terms; NaN elsewhere so averaging still works
  xs2xl_mse = xl_identity_mse = xs_identity_mse = xl_cycle_fvu = NAN
  if hasattr(head, 'up'):
    xs2xl = head.up(xs)
    xl_recon = head.up(xl2xs)
    xs2xl_mse = F.mse_loss(xs2xl, xl).item()
    xl_identity_mse = F.mse_loss(xl_recon, xl).item()
    xs_identity_mse = F.mse_loss(head.down(xs2xl), xs).item()
    xl_var = (xl - xl.mean(dim=(0, 1), keepdim=True)).pow(2).mean().item()
    xl_cycle_fvu = xl_identity_mse / xl_var if xl_var else NAN

  xs_loss = xl_loss = xl2xs_loss = NAN
  if site == 'final_post_lnf':
    xs_loss, xl_loss, xl2xs_loss = ce(xs), ce(xl), ce(xl2xs)

  return {
    'align_loss'       : 1 - matched_sim,
    'matched_sim'      : matched_sim,
    'matched_std'      : matched.std().item(),
    'centered_sim'     : centered_sim,
    'shuffled_sim'     : shuffled_sim,
    'sim_above_shuffle': matched_sim - shuffled_sim,
    'sim_above_floor'  : matched_sim - math.sqrt(2.0 / (math.pi * D)),
    'xs_scale'         : xs_scale,
    'xl_scale'         : xl_scale,
    'xs_norm'          : xs_norm,     
    'xl_norm'          : xl_norm,
    'xl2xs_norm'       : xl2xs_norm,
    'norm_ratio'       : xl2xs_norm / xs_norm if xs_norm else NAN,
    'xl2xs_mse'        : F.mse_loss(xl2xs_f, xs_f).item(),
    'xs2xl_mse'        : xs2xl_mse,
    'xl_identity_mse'  : xl_identity_mse,
    'xs_identity_mse'  : xs_identity_mse,
    'xl_cycle_fvu'     : xl_cycle_fvu,
    'xs_loss'          : xs_loss,
    'xl_loss'          : xl_loss,
    'xl2xs_loss'       : xl2xs_loss,
  }


@torch.no_grad()
def get_alignment_geometry(xs_model, xl_model, site, xs_scale, xl_scale, head, eval_batches):
  xs_model.eval(); xl_model.eval(); head.eval()
  per_batch = [_geom_one_batch(xs_model, xs_scale, xl_model, xl_scale, site, head, x, y)
               for x, y in eval_batches]
  n = len(per_batch)
  return {k: float(sum(d[k] for d in per_batch) / n) for k in per_batch[0]}


class BidirectionalHead(nn.Module):
  def __init__(self, xs_n_embed, xl_n_embed):
    super().__init__()
    self.down = nn.Linear(xl_n_embed, xs_n_embed, bias=True)
    self.up = nn.Linear(xs_n_embed, xl_n_embed, bias=True)
    nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.down.bias)
    nn.init.normal_(self.up.weight, mean=0.0, std=0.02)
    nn.init.zeros_(self.up.bias)

  def forward(self, xl):
    return self.down(xl)

def build_align_head(xs_n_embed, xl_n_embed, arch='linear_ln'):
  if arch == 'linear_ln':
      align_head = nn.Sequential(
          nn.Linear(xl_n_embed, xs_n_embed, bias=False),  # [B, T, xl_n_embed] -> [B, T, xs_nembed]
          nn.LayerNorm(xs_n_embed) # [B, T, xs_n_embed]
      ).to(device)
      nn.init.normal_(align_head[0].weight, mean=0.0, std=0.02)
      nn.init.ones_(align_head[1].weight); nn.init.zeros_(align_head[1].bias)
  elif arch == 'linear_bias':
      align_head = nn.Sequential(
          nn.Linear(xl_n_embed, xs_n_embed, bias=True),  # [B, T, xl_n_embed] -> [B, T, xs_nembed]
      ).to(device)
      nn.init.normal_(align_head[0].weight, mean=0.0, std=0.02)
      nn.init.zeros_(align_head[0].bias)
  elif arch == 'bidirectional_head':
    align_head = BidirectionalHead(xs_n_embed, xl_n_embed)
  else:
    raise ValueError(f"unknown arch: {arch}")
  return align_head.to(device)


def fit_posthoc_align_head(xs_model, xl_model, site, xs_scale, xl_scale, 
                           train_data, eval_batches, run_name, align_head_dir, 
                           config, objective='cosine', arch='linear_ln', 
                           alpha=1.0, max_steps=3000, snap_every=200,
                           snap_early_until=400, snap_early_every=10, lr=1e-3):
  final_ckpt_path = os.path.join(align_head_dir, "final.pt")
  best_ckpt_path = os.path.join(align_head_dir, "best.pt")
  traj_dir = os.path.join(align_head_dir, "traj")
  os.makedirs(traj_dir, exist_ok=True)

  xs_n_embed = xs_model.n_embed
  xl_n_embed = xl_model.n_embed
  print(f"Starting XL:{xl_n_embed} to XS:{xs_n_embed} align head training")

  align_head = build_align_head(xs_n_embed, xl_n_embed, arch=arch)

  if os.path.exists(final_ckpt_path):
    final_ck = torch.load(final_ckpt_path, map_location=device)
    snaps = sorted(os.listdir(traj_dir))
    if final_ck['step'] >= max_steps and snaps and snaps[-1] == f"step{max_steps:06d}.pt":
      ckpt = torch.load(final_ckpt_path, map_location=device)
      align_head.load_state_dict(ckpt['head'])
      print(f"Trained model available @ '{final_ckpt_path}'")
      return align_head
    print(f"Partial run at step {final_ck['step']}/{max_steps} — retraining")

  optimizer = torch.optim.AdamW(align_head.parameters(), lr=lr)
  best_loss = float('inf')

  def save(step, path, train_loss, align_geom):
    torch.save({
        'step'    : step,
        'run_name'    : run_name,
        'objective'   : objective,
        'arch'    : arch,
        'head'    : align_head.state_dict(),
        'optimizer'   : optimizer.state_dict(),
        'xs_n_embed'  : xs_n_embed,
        'xl_n_embed'  : xl_n_embed,
        'config'  : config,
        'train_loss'  : train_loss,
        'best_loss'   : best_loss,
        'align_geom'  : align_geom}, path)

  for step in range(max_steps+1):
    x, y = get_batch(train_data, config['block_size'], config['batch_size'])

    with torch.no_grad():
      xs = xs_model.forward_repr_at_site(site, x) / xs_scale
      xl = xl_model.forward_repr_at_site(site, x) / xl_scale

    if arch == 'bidirectional_head':
      xl2xs = align_head.down(xl) # down
      xs2xl = align_head.up(xs) # up
      xs_identity = align_head.down(xs2xl)
      xl_identity = align_head.up(xl2xs)
      loss = (F.mse_loss(xl2xs, xs)
              + F.mse_loss(xs2xl, xl)
              + alpha * F.mse_loss(xs_identity, xs)
              + alpha * F.mse_loss(xl_identity, xl))
    else:
      xl2xs = align_head(xl)
      if objective == 'cosine':
        loss = (1 - F.cosine_similarity(xs, xl2xs, dim=-1)).mean()
      else:
        loss = F.mse_loss(xl2xs, xs)

    optimizer.zero_grad(); loss.backward(); optimizer.step()

    is_snap = (step % snap_every == 0) or \
          (snap_early_every > 0 and step <= snap_early_until and step % snap_early_every == 0)
    if is_snap:
      align_head.eval()
      align_geom = get_alignment_geometry(xs_model, xl_model, site, xs_scale,
                                           xl_scale, align_head, eval_batches)
      align_head.train()
      str_msg = f"step {step} | train: {loss.item():.4f}, eval: {align_geom['align_loss']:.4f}"
      if best_loss > align_geom['align_loss']:
        best_loss = align_geom['align_loss']
        str_msg = str_msg + f" <- best"
        save(step, best_ckpt_path, loss.item(), align_geom)
      print(str_msg)

      # ADDED: trajectory snapshot -- head only, cheap
      torch.save({'step': step,
                  'objective' : objective,
                  'arch'      : arch,
                  'head': align_head.state_dict(),
                  'align_geom': align_geom},
                   os.path.join(traj_dir, f"step{step:06d}.pt"))
      if step >= max_steps:
        save(step, final_ckpt_path, loss.item(), align_geom)

  print(f"Finished {run_name} training; best loss: {best_loss:.4f}")
  return align_head