import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = (
  "cuda" if torch.cuda.is_available() else
  "mps" if torch.backends.mps.is_available() else
  "cpu"
)

def get_batch(data, block_size, batch_size):
  ix = torch.randint(0, (len(data) - block_size), (batch_size, )) # [B]
  x = torch.stack([data[i: i+block_size] for i in ix])            # [B, T]
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])        # [B, T]
  return x.to(device), y.to(device)

def train_standalone_probe(model, train_data, eval_batches, config,
                           run_dir, run_name,
                           max_steps=2500, eval_every=250, lr=1e-3,
                           weight_decay=0.01):
  probe_dir = os.path.join(run_dir, run_name)
  os.makedirs(probe_dir, exist_ok=True) # technically, this should already exist
  probe_path = os.path.join(probe_dir, "probe.pt")

  model.eval()
  V = config['vocab_size']
  n_embed = model.n_embed
  print(f"Sandalone probe training of n_embed: {n_embed} started...")

  # Init probe head
  probe = nn.Linear(n_embed, V, bias=False).to(device)

  if os.path.exists(probe_path):
    probe_ckpt = torch.load(probe_path, map_location=device)
    if probe_ckpt['step'] == max_steps:
      return probe.load_state_dict(probe_ckpt["probe"]), probe_ckpt["best_loss"]


  nn.init.normal_(probe.weight, mean=0.0, std=0.02)
  optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
  best_loss = float('inf')

  @torch.no_grad()
  def eval_probe():
    losses = []
    probe.eval()
    for x, y in eval_batches:
      logits = probe(model.forward_repr(x))
      B, T, _ = logits.shape
      losses.append(
          F.cross_entropy(logits.view(B*T, V), y.view(B*T)).item())
    probe.train()
    return float(np.mean(losses))

  for step in range(max_steps+1):
    x, y = get_batch(train_data, config['block_size'], config['batch_size'])
    with torch.no_grad():
      repr_ = model.forward_repr(x)
    logits = probe(repr_)
    B, T, _ = logits.shape
    loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))

    optim.zero_grad(); loss.backward(); optim.step()

    if step % eval_every == 0:
      eval_loss = eval_probe()
      str_msg = f"step {step} | train: {loss.item():.4f}, eval: {eval_loss:.4f}"
      if best_loss > eval_loss:
        best_loss = eval_loss
        str_msg = str_msg + f" <- best"
      print(str_msg)

    if step == max_steps:
      torch.save({
          "step": step,
          "n_embed": n_embed,
          "probe": probe.state_dict(),
          "best_loss": best_loss
      }, probe_path)

  print(f"Probe training complete; best loss -> {best_loss:.4f}")
  return probe, best_loss