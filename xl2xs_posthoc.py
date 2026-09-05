# @title Posthoc Alignment

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


device = (
  "cuda" if torch.cuda.is_available() else
  "mps" if torch.backends.mps.is_available() else
  "cpu"
)

def _fixed_perm(n, device, seed=4242):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g).to(device)

def get_batch(data, block_size, batch_size):
  ix = torch.randint(0, (len(data) - block_size), (batch_size, )) # [B]
  x = torch.stack([data[i: i+block_size] for i in ix])            # [B, T]
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])        # [B, T]
  return x.to(device), y.to(device)

@torch.no_grad()
def get_alignment_geometry(xs_model, xl_model, head, eval_batches,
                           block_size, batch_size):
  align_losses, matched_sims, centered_sims, shuffled_sims = [], [], [], []
  matched_stds, mses, xs_norms, xl2xs_norms = [], [], [], []
  xs_losses, xl_losses, xl2xs_losses = [], [], []

  xs_model.eval(); xl_model.eval(); head.eval()

  for x, y in eval_batches:
    xs = xs_model.forward_repr(x)
    xl = xl_model.forward_repr(x)
    xl2xs = head(xl)

    xs_logits = xs_model.lm_head(xs)
    B, T, V = xs_logits.shape
    xs_losses.append(F.cross_entropy(xs_logits.view(B*T, V), y.view(B*T)).item())
    xl_logits = xl_model.lm_head(xl)
    xl_losses.append(F.cross_entropy(xl_logits.view(B*T, V), y.view(B*T)).item())
    xl2xs_logits = xs_model.lm_head(xl2xs)
    xl2xs_losses.append(F.cross_entropy(xl2xs_logits.view(B*T, V), y.view(B*T)).item())

    xs_flat = xs.reshape(-1, xs.shape[-1])
    xl2xs_flat = xl2xs.reshape(-1, xl2xs.shape[-1])

    matched = F.cosine_similarity(xs_flat, xl2xs_flat, dim=-1)
    align_losses.append(1 - matched.mean().item())
    matched_sims.append(matched.mean().item())
    matched_stds.append(matched.std().item())

    perm = _fixed_perm(xs_flat.shape[0], xs_flat.device)
    shuffled_sim = F.cosine_similarity(xs_flat[perm], xl2xs_flat, dim=-1).mean().item()
    shuffled_sims.append(shuffled_sim)

    centered_sim = F.cosine_similarity(
        xs_flat - xs_flat.mean(dim=0, keepdim=True),
        xl2xs_flat - xl2xs_flat.mean(dim=0, keepdim=True), dim=-1).mean().item()
    centered_sims.append(centered_sim)

    D = xs_model.n_embed
    xs_norms.append(xs_flat.norm(dim=-1).mean().item())
    xl2xs_norms.append(xl2xs_flat.norm(dim=-1).mean().item())
    mses.append(F.mse_loss(xs_flat, xl2xs_flat).item())
    xs_norm    = sum(xs_norms) / len(xs_norms)
    xl2xs_norm = sum(xl2xs_norms) / len(xl2xs_norms)
    matched    = sum(matched_sims) / len(matched_sims)

  return {
      'align_loss'    : sum(align_losses) / len(align_losses),
      'matched_sim'   : matched,
      'matched_std'   : sum(matched_stds) / len(matched_stds),
      'centered_sim'  : sum(centered_sims) / len(centered_sims),
      'shuffled_sim'  : sum(shuffled_sims) / len(shuffled_sims),
      'xs_norm' : xs_norm,
      'xl2xs_norm'  : xl2xs_norm,
      'mse'  : sum(mses) / len(mses),
      'xs_loss': sum(xs_losses) / len(xs_losses),
      'xl_loss': sum(xl_losses) / len(xl_losses),
      'xl2xs_via_xs_loss':  sum(xl2xs_losses) / len(xl2xs_losses),
      # E|cos| for two random D-dim vectors = sqrt(2/(pi*D)); indicating
      # reducing XS size improves consine could be an artifact of this.
      'random_floor_sim': math.sqrt(2.0 / (math.pi * D)),
      'sim_above_floor' : matched - math.sqrt(2.0 / (math.pi * D)),
      # different scale from XS matters directly for the SAE stage.
      'norm_ratio': xl2xs_norm / xs_norm if xs_norm else float('nan'),
  }

def train_xl2xs_probe(xs_model, xl_model, head, train_data, eval_batches,
                      config, steps=3000, eval_every=250,
                      lr=1e-3, weight_decay=0.01):
  xs_n_embed = xs_model.n_embed
  xl_n_embed = xl_model.n_embed
  V = config['vocab_size']
  print(f"Starting XL:{xl_n_embed} to XS:{xs_n_embed} probe training...")

  #init probe
  probe = nn.Linear(xs_n_embed, V, bias=False).to(device)
  nn.init.normal_(probe.weight, mean=0.0, std=0.02)
  optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
  xl_model.eval(); head.eval()

  @torch.no_grad()
  def eval_probe():
    losses = []
    probe.eval()
    for x, y in eval_batches:
      B, T = y.shape
      xl = xl_model.forward_repr(x)
      xl2xs = head(xl)
      loss = F.cross_entropy(probe(xl2xs).view(B*T, V), y.view(B*T))
      losses.append(loss.item())
    probe.train()
    return sum(losses) / len(losses)

  best_loss = float('inf')
  for step in range(steps+1):
    x, y = get_batch(train_data, config['block_size'], config['batch_size'])

    # forward
    with torch.no_grad():
      xl2xs = head(xl_model.forward_repr(x))
    logits = probe(xl2xs)
    B, T, _ = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))

    #backward
    optim.zero_grad(); loss.backward(); optim.step()

    if step % eval_every == 0:
      eval_loss = eval_probe()
      str_msg = f"step {step} | train: {loss.item():.4f}, eval: {eval_loss:.4f}"
      if best_loss > eval_loss:
        best_loss = eval_loss
        str_msg = str_msg + f" <- best"
      print(str_msg)
  print(f"Finished XL:{xl_n_embed} to XS:{xs_n_embed} probe training; "
        f"best loss {best_loss:.4f}")
  return probe, best_loss


def fit_posthoc_align_head(xs_model, xl_model, train_data, eval_batches,
                           run_name, run_dir, config,
                           steps=3000, eval_every=200, lr=1e-3,
                           traj_every=None):
  os.makedirs(run_dir, exist_ok=True)
  final_run_dir = os.path.join(run_dir, run_name)
  os.makedirs(final_run_dir, exist_ok=True)
  final_ckpt_path = os.path.join(final_run_dir, "final.pt")
  best_ckpt_path = os.path.join(final_run_dir, "best.pt")
  traj_dir = os.path.join(final_run_dir, "traj")
  if traj_every:
      os.makedirs(traj_dir, exist_ok=True)

  xs_n_embed = xs_model.n_embed
  xl_n_embed = xl_model.n_embed
  print(f"Starting XL:{xl_n_embed} to XS:{xs_n_embed} align head training")

  # Init Align head
  align_head = nn.Sequential(
      nn.Linear(xl_n_embed, xs_n_embed, bias=False),  # [B, T, xl_n_embed] -> [B, T, xs_nembed]
      nn.LayerNorm(xs_n_embed) # [B, T, xs_n_embed]
  ).to(device)

  nn.init.normal_(align_head[0].weight, mean=0.0, std=0.02)
  nn.init.ones_(align_head[1].weight); nn.init.zeros_(align_head[1].bias)

  if os.path.exists(final_ckpt_path):
    final_ck = torch.load(final_ckpt_path, map_location=device)
    if final_ck['step'] >= steps:
      ckpt = torch.load(best_ckpt_path, map_location=device)
      align_head.load_state_dict(ckpt['head'])
      print(f"Trained model available @ '{best_ckpt_path}' "
            f"with best loss {ckpt['best_loss']:.4f}")
      return align_head
    print(f"Partial run at step {final_ck['step']}/{steps} — retraining")

  optimizer = torch.optim.AdamW(align_head.parameters(), lr=lr)
  best_loss = float('inf')

  def save(step, path, train_loss, align_geom):
    torch.save({
        'step'    : step,
        'run_name'    : run_name,
        'head'    : align_head.state_dict(),
        'optimizer'   : optimizer.state_dict(),
        'xs_n_embed'  : xs_n_embed,
        'xl_n_embed'  : xl_n_embed,
        'config'  : config,
        'train_loss'  : train_loss,
        'best_loss'   : best_loss,
        'align_geom'  : align_geom}, path)

  for step in range(steps+1):
    x, y = get_batch(train_data, config['block_size'], config['batch_size'])

    with torch.no_grad():
      xs = xs_model.forward_repr(x)
      xl = xl_model.forward_repr(x)

    xl2xs = align_head(xl)
    loss = (1 - F.cosine_similarity(xs, xl2xs, dim=-1)).mean()

    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if step % eval_every == 0:
      align_head.eval()
      align_geom = get_alignment_geometry(xs_model, xl_model, align_head,
                                          eval_batches, config['block_size'],
                                          config['batch_size'])
      align_head.train()
      str_msg = f"step {step} | train: {loss.item():.4f}, eval: {align_geom['align_loss']:.4f}"
      if best_loss > align_geom['align_loss']:
        best_loss = align_geom['align_loss']
        str_msg = str_msg + f" <- best"
        save(step, best_ckpt_path, loss.item(), align_geom)
      print(str_msg)
      save(step, final_ckpt_path, loss.item(), align_geom)

      # ADDED: trajectory snapshot -- head only, cheap
      if traj_every and step % traj_every == 0:
        torch.save({'step': step, 'head': align_head.state_dict(),
                    'align_geom': align_geom},
                   os.path.join(traj_dir, f"step{step:06d}.pt"))

  print(f"Finished {run_name} training; best loss: {best_loss:.4f}")
  return align_head