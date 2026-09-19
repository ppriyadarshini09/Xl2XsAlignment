import os
import torch
import numpy as np
import gc, random
from model import GPT
from train_gpt import device, compute_scale, eval_loss
from xl2xs_posthoc import fit_posthoc_align_head, build_align_head
from train_probes import train_xl2xs_probe, train_standalone_probe


# --------------- Utils ----------------
def get_batch(data, block_size, batch_size):
    ix = torch.randint(0, (len(data) - block_size), (batch_size,))  # [B]
    x = torch.stack([data[i : i + block_size] for i in ix])  # [B, T]
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # [B, T]
    return x.to(device), y.to(device)

def make_eval_batches(data, config, n_batches=10, eval_seed=999):
    state = torch.get_rng_state()
    torch.manual_seed(eval_seed)
    batches = [
        get_batch(data, config["block_size"], config["batch_size"])
        for _ in range(n_batches)
    ]
    torch.set_rng_state(state)
    return batches

def free(*objs):
  for o in objs:
    del o
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

def set_seed(s):
  random.seed(s)
  np.random.seed(s)
  torch.manual_seed(s)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(s)

# ---------------- Load Model ----------------
def load_model(bank_path, config, eval_batches, freeze=True):
  """Load banked model."""
  banked_meta = torch.load(bank_path, map_location=device)
  meta = banked_meta['meta']
  role, n_embed, seed = meta['role'], meta['n_embed'], meta['seed']

  model = GPT(
      vocab_size=config['vocab_size'],
      block_size=config['block_size'],
      n_embed=n_embed,
      n_heads=config['n_heads'],
      n_layers=config['n_layers'],
      dropout=config['dropout']
  ).to(device)
  model.load_state_dict(banked_meta['model'])
  model.eval()

  if freeze:
    for p in model.parameters():
      p.requires_grad_(False)

  loss_again = eval_loss(model, eval_batches)
  assert abs(loss_again - meta['eval_loss'] < 1e-4), (
      f"{role}_{size}_s{seed} load mismatch: banked loss {meta['eval_loss']:.6f}"
      f" vs recalculated loss {loss_again:.6f} - model was mis-configured while"
      f" loading (check initialization, dropouts etc)"
  )
  return model, meta

# ---------------- Run Alignhead Trajectory Sweep ----------------
def align_head_trajetory_seed_sweep(traj_seeds, traj_config, model_bank_dir,
                                    base_align_head_dir, train_data, eval_batches,
                                    train_config,
                                    objective='mse', arch='bidirectional_head',
                                    site='final_pre_lnf', scaled=False, 
                                    alpha=1.0):
  for traj_seed in traj_seeds:
    traj_xs = traj_config['traj_xs']
    traj_xl = traj_config['traj_xl']
    traj_run = f"traj_xl{traj_xl}_xs{traj_xs}_s{traj_seed}"
    align_head_dir = os.path.join(base_align_head_dir, traj_run, site)
    os.makedirs(align_head_dir, exist_ok=True)
    align_head_run_dir = os.path.join(align_head_dir, f"{objective}_{arch}")

    if arch == 'bidirectional_head':
      alpha_name = f"{alpha:g}".replace('.', 'p').replace('-', 'm')
      align_head_run_dir = os.path.join(align_head_run_dir, f"alpha_{alpha_name}")
    os.makedirs(align_head_run_dir, exist_ok=True)

    print(f"\n Starting {traj_run}" \
          f"with align head objective: {objective} & arch: {arch}")
    traj_dir = os.path.join(align_head_run_dir, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    align_head_final_path = os.path.join(align_head_run_dir, "final.pt")
    traj_stats_csv = os.path.join(align_head_run_dir, "trajectory.csv")

    if os.path.exists(traj_stats_csv):
      print(f"Entire trajectory run for {traj_run} (obj:{objective}," \
            f"arch:{arch}) already available.")
      continue
    else:
      print(f"Trajectory run for {traj_run} (obj:{objective}, arch:{arch})" \
            f" was not started or partially done. Starting again...")

    # Load XS model, compute XS scale
    xs_bank_path = os.path.join(model_bank_dir, f"xs_{traj_xs}_s{traj_seed}.pt")
    xs_model, _ = load_model(xs_bank_path, train_config, eval_batches)
    xs_scale = compute_scale(xs_model, site, eval_batches) if scaled else 1.0
    # Load XL model, compute XL scale
    xl_bank_path = os.path.join(model_bank_dir, f"xl_{traj_xl}_s{traj_seed}.pt")
    xl_model, _ = load_model(xl_bank_path, train_config, eval_batches)
    xl_scale = compute_scale(xl_model, site, eval_batches) if scaled else 1.0

    align_head_probe_dir = os.path.join(align_head_dir, "probes")
    os.makedirs(align_head_probe_dir, exist_ok=True)
    set_seed(traj_config['probe_seed'])
    _, xs_probe_loss = train_standalone_probe(
        xs_model, site, xs_scale, train_data, eval_batches, train_config,
        run_name=traj_run, probe_dir=align_head_probe_dir, max_steps=traj_config['probe_steps'])
    set_seed(traj_config['probe_seed'])
    _, xl_probe_loss = train_standalone_probe(
        xl_model, site, xl_scale, train_data, eval_batches, train_config,
        run_name=traj_run, probe_dir=align_head_probe_dir, max_steps=traj_config['probe_steps'])
    gap = xs_probe_loss - xl_probe_loss
    print(f"\nXS {xs_probe_loss:.4f} | XL {xl_probe_loss:.4f} | gap {gap:.4f} nats\n")
    assert gap > 0, "XL does not beat XS -- retention is undefined for this pair"

    set_seed(traj_config['probe_seed'])
    _ = fit_posthoc_align_head(
        xs_model=xs_model, xl_model=xl_model, site=site,
        xs_scale=xs_scale, xl_scale=xl_scale,
        train_data=train_data, eval_batches=eval_batches,
        run_name=traj_run, align_head_dir=align_head_run_dir,
        config=train_config,
        objective=objective,
        arch=arch,
        max_steps=traj_config['align_steps'],
        snap_every=traj_config['snap_every'],
        snap_early_until=traj_config['snap_early_until'],
        snap_early_every=traj_config['snap_early_every'],
        alpha=alpha,
        lr=traj_config['lr'])

    # ---- walk the snapshots, probe each one ----
    rows = []
    traj_probe_dir = os.path.join(align_head_run_dir, "traj_probes")
    os.makedirs(traj_probe_dir, exist_ok=True)
    for fn in sorted(os.listdir(traj_dir)):
        snap = torch.load(os.path.join(traj_dir, fn), map_location=device)

        head = build_align_head(traj_xs, traj_xl, arch=snap['arch'])
        head.load_state_dict(snap['head'])
        head.eval()

        set_seed(traj_config['probe_seed'])
        _, probe_loss = train_xl2xs_probe(
            xs_model, xl_model, site, xs_scale, xl_scale, head, train_data, 
            eval_batches, train_config, steps=traj_config['probe_steps'], 
            traj_probe_dir=traj_probe_dir, traj_step_pt=fn)

        g = snap['align_geom']
        rows.append({'step': snap['step'],
                     **g,
                     'xl_probe_loss': xl_probe_loss,
                     'xs_probe_loss': xs_probe_loss,
                     'gap_nats': gap,
                     'xl2xs_probe_loss': probe_loss,
                     'retained': (xs_probe_loss - probe_loss) / gap,
                     })
        print(f"  step {snap['step']:5d} | sim {g['sim_above_shuffle']:.4f} | "
              f"retained {rows[-1]['retained']:.1%}")
        free(head)

    traj_df = pd.DataFrame(rows)
    traj_df['seed'] = traj_seed
    traj_df['objective'] = objective
    traj_df['arch'] = arch
    traj_df['scaled'] = scaled
    traj_df['alpha'] = alpha
    traj_df.to_csv(traj_stats_csv, index=False)

    # ---- the test ----
    fit_df = traj_df[traj_df['step'] > 0]
    r = fit_df['sim_above_shuffle'].corr(fit_df['retained'], method='spearman')
    print(f"\nSpearman(cosine, retained) along the fit = {r:+.3f}")
    print(f"  cosine  {fit_df['sim_above_shuffle'].iloc[0]:.4f} -> "
          f"{fit_df['sim_above_shuffle'].iloc[-1]:.4f}")
    print(f"  retained {fit_df['retained'].iloc[0]:.1%} -> "
          f"{fit_df['retained'].iloc[-1]:.1%}")
    if r < -0.5:
        print("\n-> INVERSION at fixed D. No dimensional artifact can explain this.\n")
    elif r > 0.5:
        print("\n-> Cosine and retention move TOGETHER at fixed D. The width-sweep "
              "inversion is then most likely a dimensional artifact.\n")
    else:
        print("\n-> No clear within-fit relationship. The width sweep is doing the "
              "work, and the dimension confound is not ruled out.\n")
    free(xl_model); free(xs_model)