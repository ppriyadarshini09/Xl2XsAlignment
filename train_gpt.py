import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import GPT

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"using device: {device}")


## ---------------- Utils ----------------
def make_eval_batches(data, config, n_batches=10, eval_seed=999):
    state = torch.get_rng_state()
    torch.manual_seed(eval_seed)
    batches = [
        get_batch(data, config["block_size"], config["batch_size"])
        for _ in range(n_batches)
    ]
    torch.set_rng_state(state)
    return batches


@torch.no_grad()
def eval_loss(model, batches=None):
  assert batches is not None, \
    "batches to compute eval loss can not be None"
  was_training = model.training
  model.eval()
  losses = [model(x, y)[1].item() for x, y in batches]
  if was_training:
    model.train()
  return float(np.mean(losses))


def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, (len(data) - block_size), (batch_size,))  # [B]
    x = torch.stack([data[i : i + block_size] for i in ix])  # [B, T]
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # [B, T]
    return x.to(device), y.to(device)


## ---------------- Train GPT ----------------
def train_standalone(train_data, eval_batches, config, n_embed, run_dir, run_name):
  os.makedirs(run_dir, exist_ok=True)
  final_dir = os.path.join(run_dir, run_name)
  os.makedirs(final_dir, exist_ok=True)
  final_ckpt_path = os.path.join(final_dir, "final.pt")
  best_ckpt_path = os.path.join(final_dir, "best.pt")
  metrics_path = os.path.join(final_dir, "metrics.csv")

  model = GPT(
      vocab_size=config['vocab_size'],
      block_size=config['block_size'],
      n_embed=n_embed,
      n_heads=config['n_heads'],
      n_layers=config['n_layers'],
      dropout=config['dropout']
  ).to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

  best_loss = float('inf')
  best_step = 0
  stale = 0
  stopped_early = False

  patience   = config.get('patience', 6)
  min_delta  = config.get('min_delta', 1e-3)
  min_steps  = int(config['max_steps'] * config.get('min_steps_frac', 0.55))
  enabled    = config.get('early_stop', True)

  def save(path, step, loss):
    torch.save({
        'run_name': run_name,
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'n_embed': n_embed,
        'eval_loss': loss,
        # audit trail: needed to check XS and XL stopped comparably
        'stopped_early': stopped_early,
        'stop_step': step,
        'max_steps': config['max_steps'],
    }, path)

  for step in range(config['max_steps']+1):
    x, y = get_batch(train_data, config['block_size'], config['batch_size'])
    logits, loss = model(x, y)

    # backward pass
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    optimizer.step()

    if step % config['eval_every'] == 0:
      ev = eval_loss(model, eval_batches)
      str_msg = f"step {step:5d} | train: {loss.item():.4f}, eval: {ev:.4f}"

      # ---- best tracking and staleness are SEPARATE questions ----
      # any new minimum gets checkpointed; only a min_delta-sized one
      # resets the patience counter.
      improved_really = ev < best_loss - min_delta
      if ev < best_loss:
        best_loss, best_step = ev, step
        str_msg += " <- best"
        save(best_ckpt_path, step, best_loss)
      stale = 0 if improved_really else stale + 1

      if stale > 0:
        str_msg += f"  (stale {stale}/{patience})"
      print(str_msg)

      if enabled and stale >= patience and step >= min_steps:
        stopped_early = True
        print(f"\nearly stop @ step {step}: no >{min_delta} improvement for "
              f"{patience} evals. best {best_loss:.4f} @ step {best_step} "
              f"({step/config['max_steps']:.0%} of schedule, lr {config['lr']:.2e})")
        break

      if enabled and stale >= patience and step < min_steps:
        # plateaued but the LR hasn't decayed yet — keep going, and say why
        print(f"     plateaued at {step/config['max_steps']:.0%} of schedule; "
              f"holding until {min_steps} so the LR decay still happens")

  save(final_ckpt_path, step, ev)

  print(f"Training complete; best loss -> {best_loss:.4f}")
  print(f"  best  -> {best_ckpt_path}")
  print(f"  final -> {final_ckpt_path}")
  return model