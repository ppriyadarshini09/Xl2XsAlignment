import os
import shutil
import torch
import numpy as np
import pandas as pd
import gc, random
from train_gpt import device, compute_scale, eval_loss, GPT, finish_forward
from xl2xs_posthoc import fit_posthoc_align_head, build_align_head, get_alignment_geometry
from train_probes import train_xl2xs_probe, train_standalone_probe, probe_ce


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
      f"{role}_{n_embed}_s{seed} load mismatch: banked loss {meta['eval_loss']:.6f}"
      f" vs recalculated loss {loss_again:.6f} - model was mis-configured while"
      f" loading (check initialization, dropouts etc)"
  )
  return model, meta

def recompute_traj_metrics(traj_seeds, traj_config, model_bank_dir,
                           base_align_head_dir, eval_batches, train_config,
                           objective='mse', arch='bidirectional_head',
                           site='final_pre_lnf', scaled=False, alpha=1.0,
                           write_csv=True):
  V = train_config['vocab_size']
  for traj_seed in traj_seeds:
    traj_xs = traj_config['traj_xs']
    traj_xl = traj_config['traj_xl']
    traj_run = f"traj_xl{traj_xl}_xs{traj_xs}_s{traj_seed}"
    align_head_dir = os.path.join(base_align_head_dir, traj_run, site)
    align_head_run = f"{objective}_{arch}_scaled" if scaled else f"{objective}_{arch}_unscaled"
    align_head_run_dir = os.path.join(align_head_dir, align_head_run)

    if arch == 'bidirectional_head':
      alpha_name = f"{alpha:g}".replace('.', 'p').replace('-', 'm')
      align_head_run_dir = os.path.join(align_head_run_dir, f"alpha_{alpha_name}")

    print(f"\n Loading from {align_head_run_dir} for recomputation...")
    traj_dir = os.path.join(align_head_run_dir, "traj")
    align_head_probe_dir = os.path.join(align_head_run_dir, "probes")

    xs_rep = lambda x: xs_model.forward_repr_at_site(site, x) / xs_scale
    xl_rep = lambda x: xl_model.forward_repr_at_site(site, x) / xl_scale

    # Load XS model and probe, compute XS scale
    xs_bank_path = os.path.join(model_bank_dir, f"xs_{traj_xs}_s{traj_seed}.pt")
    xs_model, _ = load_model(xs_bank_path, train_config, eval_batches)
    xs_scale = compute_scale(xs_model, site, eval_batches) if scaled else 1.0

    xs_probe_ckpt = torch.load(
        os.path.join(align_head_probe_dir, f"probe_{traj_xs}.pt"),
        map_location=device)
    xs_probe = nn.Linear(traj_xs, V, bias=False).to(device)
    xs_probe.load_state_dict(xs_probe_ckpt['probe'])
    xs_chk = xs_probe_ckpt['best_loss']

    # Load XL model and probe, compute XL scale
    xl_bank_path = os.path.join(model_bank_dir, f"xl_{traj_xl}_s{traj_seed}.pt")
    xl_model, _ = load_model(xl_bank_path, train_config, eval_batches)
    xl_scale = compute_scale(xl_model, site, eval_batches) if scaled else 1.0

    xl_probe_ckpt = torch.load(
        os.path.join(align_head_probe_dir, f"probe_{traj_xl}.pt"),
        map_location=device)
    xl_probe = nn.Linear(traj_xl, V, bias=False).to(device)
    xl_probe.load_state_dict(xl_probe_ckpt['probe'])
    xl_chk = xl_probe_ckpt['best_loss']

    # Sanity: the cached probes must match THIS scaling convention.
    # (Training evals were monotone, so best_loss == loss of the saved weights.)
    xs_probe_loss = probe_ce(xs_probe, xs_rep, eval_batches)
    xl_probe_loss = probe_ce(xl_probe, xl_rep, eval_batches)
    probe_tol = 1e-3
    if abs(xs_chk - xs_probe_loss) > probe_tol or abs(xl_chk - xl_probe_loss) > probe_tol:
        raise RuntimeError(
            f"cached probes don't match this site/scaling: "
            f"xs {xs_chk:.4f} vs {xs_probe_loss:.4f}, xl {xl_chk:.4f} vs {xl_probe_loss:.4f}. "
            f"Wrong probe_dir or scaled flag?")

    gap = xs_probe_loss - xl_probe_loss
    print(f"\nXS {xs_probe_loss:.4f} | XL {xl_probe_loss:.4f} | gap {gap:.4f} nats\n")
    assert gap > 0, "XL does not beat XS -- retention is undefined for this pair"

    traj_probe_dir = os.path.join(align_head_run_dir, "traj_probes")
    assert os.path.exists(traj_probe_dir), \
      f"Traj probe directory not found @ {traj_probe_dir}"

    traj_dir = os.path.join(align_head_run_dir, "traj")
    rows = []
    for fn in sorted(os.listdir(traj_probe_dir)):

      # get xl2xs traj probe, to compute retrained funtional transfer metric
      probe_snap = torch.load(os.path.join(traj_probe_dir, fn), map_location=device)
      xl2xs_probe = nn.Linear(traj_xs, V, bias=False).to(device)
      xl2xs_probe.load_state_dict(probe_snap['probe'])

      align_head_traj = os.path.join(traj_dir, fn)
      assert os.path.exists(align_head_traj), \
        f"align head snapshot doesn't exist @ {align_head_traj}"

      # get xl2xs align head, to compute frozen functional transfer & geometric
      # alignment metric.
      traj_snap = torch.load(os.path.join(traj_dir, fn), map_location=device)

      bridge = build_align_head(traj_xs, traj_xl, arch=arch)
      bridge.load_state_dict(traj_snap['head'])
      bridge.eval()
      bridged = lambda x: bridge(xl_rep(x))
      xl2xs_probe_loss = probe_ce(xl2xs_probe, bridged, eval_batches)
      xl2xs_via_xs_probe_loss = probe_ce(xs_probe, bridged, eval_batches)
      align_geom = get_alignment_geometry(xs_model, xl_model, site, xs_scale,
                                           xl_scale, bridge, eval_batches)
      rows.append({'step': traj_snap['step'], **align_geom,
                   'xl_probe_loss': xl_probe_loss,
                   'xs_probe_loss': xs_probe_loss,
                   'gap_nats': gap,
                   'xl2xs_probe_loss': xl2xs_probe_loss,
                   'retained': (xs_probe_loss - xl2xs_probe_loss) / gap,
                   'frozen_retained': (xs_probe_loss - xl2xs_via_xs_probe_loss) / gap,
                   })
      print(f"  step {traj_snap['step']:5d} | sim {align_geom['sim_above_shuffle']:.4f} | "
              f"retained {rows[-1]['retained']:6.1%} | frozen {rows[-1]['frozen_retained']:7.1%} | "
              f"stitched {rows[-1]['stitched_retained']:7.1%}")
      del bridge
      
    df = pd.DataFrame(rows)
    df['seed'], df['objective'], df['arch'] = traj_seed, objective, arch
    df['site'], df['scaled'], df['alpha'] = site, scaled, alpha
    df['xs_scale'], df['xl_scale'] = xs_scale, xl_scale
 
    if write_csv:
        csv_path = os.path.join(align_head_run_dir, 'trajectory.csv')
        backup = os.path.join(align_head_run_dir, 'trajectory_orig.csv')
        if os.path.exists(csv_path) and not os.path.exists(backup):
            shutil.copy(csv_path, backup)          # keep the first original only
        df.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}" + (f" (original kept at {backup})"
                                     if os.path.exists(backup) else ""))

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
    align_head_run = f"{objective}_{arch}_scaled" if scaled else f"{objective}_{arch}_unscaled"
    align_head_run_dir = os.path.join(align_head_dir, align_head_run)

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

    align_head_probe_dir = os.path.join(align_head_run_dir, "probes")
    os.makedirs(align_head_probe_dir, exist_ok=True)
    set_seed(traj_config['probe_seed'])
    _, xs_probe_loss = train_standalone_probe(
        xs_model, site, xs_scale, train_data, eval_batches, train_config,
        probe_dir=align_head_probe_dir, run_name=traj_run, max_steps=traj_config['probe_steps'])
    set_seed(traj_config['probe_seed'])
    _, xl_probe_loss = train_standalone_probe(
        xl_model, site, xl_scale, train_data, eval_batches, train_config,
        probe_dir=align_head_probe_dir, run_name=traj_run, max_steps=traj_config['probe_steps'])
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
    traj_probe_dir = os.path.join(align_head_run_dir, "traj_probes")
    os.makedirs(traj_probe_dir, exist_ok=True)
    for fn in sorted(os.listdir(traj_dir)):
        snap = torch.load(os.path.join(traj_dir, fn), map_location=device)

        head = build_align_head(traj_xs, traj_xl, arch=snap['arch'])
        head.load_state_dict(snap['head'])
        head.eval()

        set_seed(traj_config['probe_seed'])
        _, xl2xs_probe_loss = train_xl2xs_probe(
            xs_model, xl_model, site, xs_scale, xl_scale, head, train_data, 
            eval_batches, train_config, steps=traj_config['probe_steps'], 
            traj_probe_dir=traj_probe_dir, traj_step_pt=fn)
        free(head)
    free(xl_model); free(xs_model)
    
    # ---------------- metrics phase (load-only, same code as recompute) ----------------
    traj_df = recompute_trajectory_metrics(
        arm_dir=run_dir, probe_dir=probe_dir, traj_seed=traj_seed,
        traj_config=traj_config, model_bank_dir=model_bank_dir,
        eval_batches=eval_batches, train_config=train_config,
        site=site, scaled=scaled, objective=objective, arch=arch, alpha=alpha)

    fit = traj_df[traj_df.step > 0]
    pk = traj_df.loc[traj_df.retained.idxmax()]
    print(f"\npeak {pk.retained:.1%} @ {int(pk.step)} | end {traj_df.retained.iloc[-1]:.1%} | "
          f"spearman(sim, retained) {fit.sim_above_shuffle.corr(fit.retained, method='spearman'):+.3f}")
