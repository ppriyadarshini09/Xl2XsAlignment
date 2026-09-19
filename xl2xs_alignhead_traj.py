import os
import torch
import numpy as np
import gc, random
from train_gpt import device, compute_scale, train_standalone_probe
from xl2xs_posthoc import fit_posthoc_align_head
from train_probes import train_xl2xs_probe, train_standalone_probe

SEEDS = [0, 1, 2]
PROBE_SEED = 1234
TRAJ_XS, TRAJ_XL = 64, 512
PROBE_STEPS = 2500
ALIGN_STEPS = 2000
SNAP_EVERY = 200
SNAP_EVERY_EARLY  = 20
SNAP_EARLY_UNTIL  = 400
TRAJ_SEEDS = [0] # instead of [0, 1, 2]

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

def align_head_trajetory_seed_sweep(base_align_head_dir, objective, arch,
                                    site='final_pre_lnf', scaled=False, 
                                    alpha=1.0, lr=1e-3):
  for traj_seed in TRAJ_SEEDS:
    traj_run = f"traj_xl{TRAJ_XL}_xs{TRAJ_XS}_s{traj_seed}"
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

    xs_model, _ = load_model("xs", TRAJ_XS, traj_seed, train_config, EVAL_BATCHES)
    xs_scale = compute_scale(xs_model, site, EVAL_BATCHES) if scaled else 1.0
    xl_model, _ = load_model("xl", TRAJ_XL, traj_seed, train_config, EVAL_BATCHES)
    xl_scale = compute_scale(xl_model, site, EVAL_BATCHES) if scaled else 1.0

    align_head_probe_dir = os.path.join(align_head_dir, "probes")
    os.makedirs(align_head_probe_dir, exist_ok=True)
    set_seed(PROBE_SEED)
    _, xs_probe_loss = train_standalone_probe(
        xs_model, site, xs_scale, train_data, EVAL_BATCHES, train_config,
        run_name=traj_run, probe_dir=align_head_probe_dir, max_steps=PROBE_STEPS)
    set_seed(PROBE_SEED)
    _, xl_probe_loss = train_standalone_probe(
        xl_model, site, xl_scale, train_data, EVAL_BATCHES, train_config,
        run_name=traj_run, probe_dir=align_head_probe_dir, max_steps=PROBE_STEPS)
    gap = xs_probe_loss - xl_probe_loss
    print(f"\nXS {xs_probe_loss:.4f} | XL {xl_probe_loss:.4f} | gap {gap:.4f} nats\n")
    assert gap > 0, "XL does not beat XS -- retention is undefined for this pair"

    set_seed(PROBE_SEED)
    _ = fit_posthoc_align_head(
        xs_model=xs_model, xl_model=xl_model, site=site,
        xs_scale=xs_scale, xl_scale=xl_scale,
        train_data=train_data, eval_batches=EVAL_BATCHES,
        run_name=traj_run, align_head_dir=align_head_run_dir,
        config=train_config,
        objective=objective,
        arch=arch,
        max_steps=ALIGN_STEPS,
        snap_every=SNAP_EVERY,
        snap_early_until=SNAP_EARLY_UNTIL,
        snap_early_every=SNAP_EVERY_EARLY,
        alpha=alpha,
        lr=lr)

    # ---- walk the snapshots, probe each one ----
    rows = []
    traj_probe_dir = os.path.join(align_head_run_dir, "traj_probes")
    os.makedirs(traj_probe_dir, exist_ok=True)
    for fn in sorted(os.listdir(traj_dir)):
        snap = torch.load(os.path.join(traj_dir, fn), map_location=device)

        head = build_align_head(TRAJ_XS, TRAJ_XL, arch=snap['arch'])
        head.load_state_dict(snap['head'])
        head.eval()

        set_seed(PROBE_SEED)
        _, probe_loss = train_xl2xs_probe(
            xs_model, xl_model, site, xs_scale, xl_scale, head, train_data, 
            EVAL_BATCHES, train_config, steps=PROBE_STEPS, 
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