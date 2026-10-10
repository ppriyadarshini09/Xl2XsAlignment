# @title Eval plot libraries
import math
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PAPER_STYLE = {
    'figure.dpi': 140,
    'savefig.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#e6e6e6',
    'grid.linewidth': 0.6,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'lines.linewidth': 1.6,
    'lines.markersize': 2.5,
}

# muted, colorblind-safe, distinguishable in greyscale
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']


# ---------------------------------------------------------------------------
# Summary table spec
# ---------------------------------------------------------------------------
# One row per (group, label, key, fmt, better[, step_key]):
# group    : top-level header (columns with the same group are merged)
# label    : sub-header shown under the group
# key      : metric key produced by trajectory_metrics()
# fmt      : format string for the cell
# better   : 'high' / 'low' -> shaded green where favourable; None -> no shading
# step_key : optional; appends " @ <step>" to the cell (shading still uses `key`)
# Columns that are NaN for every row (e.g. FVU for non-bidirectional arms) are dropped.
PCT, FLT, INT, SCI = '{:.1%}', '{:.3f}', '{:d}', '{:.2e}'

SUMMARY_SPEC = [
    ('Retrained',      'peak @ step',      'r_peak',       PCT, 'high', 'r_peak_step'),
    ('Retrained',      'end',              'r_end',        PCT, 'high'),
    ('Retrained',      'drop',             'r_drop',       PCT, 'low'),
    ('Frozen',         'peak @ step',      'f_peak',       PCT, 'high', 'f_peak_step'),
    ('Frozen',         'end',              'f_end',        PCT, 'high'),
    ('Frozen',         'drop',             'f_drop',       PCT, 'low'),
    ('Stitched',       'peak @ step',      's_peak',       PCT, 'high', 's_peak_step'),
    ('Stitched',       'end',              's_end',        PCT, 'high'),
    ('Stitched',       'drop',             's_drop',       PCT, 'low'),
    ('Sim > shuffle',  '@ retrained peak', 'sim',          PCT, 'high'),
    ('Sim > shuffle',  'end',              'sim_end',      PCT, 'high'),
    ('Centered sim',   '@ retrained peak', 'centered',     PCT, 'high'),
    ('Centered sim',   'end',              'centered_end', PCT, 'high'),
    ('FVU (end)',      'XL cycle',         'xl_cycle_fvu', FLT, 'low'),
    ('FVU (end)',      'XS→XL',            'xs2xl_fvu',    FLT, 'low'),
    ('Variance (end)', 'XL',               'xl_var',       SCI, None),
    ('Variance (end)', 'XS',               'xs_var',       SCI, None),
]


def _isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _cell(v, fmt, step=None, sd=None):
    """Format one cell; NaN -> em dash, optional ' ± sd' and ' @ step' suffixes."""
    if _isnan(v):
        return '—'
    if sd is not None and not _isnan(sd):
        s = (f'{v * 100:.1f} ± {sd * 100:.1f}%' if fmt == PCT
             else f'{fmt.format(v)} ± {fmt.format(sd)}')
    else:
        s = fmt.format(int(v)) if fmt == INT else fmt.format(v)
    if step is not None and not _isnan(step):
        s += f' @ {int(step)}'
    return s


def summary_table(rows, spec, caption=None, cmap='Greens', strength=0.45):
    """Turn a list of per-run dicts into a shaded pandas Styler.

    Cells are pre-formatted strings (so values like "71.0% @ 1100" are possible);
    shading and bolding are driven by the underlying numeric value via `gmap`.
    Favourable cells get a darker green, the best per column is bolded, and
    columns with better=None stay plain.
    """
    raw = pd.DataFrame(rows).set_index('name')
    raw.index.name = None
    cols = pd.MultiIndex.from_tuples([(s[0], s[1]) for s in spec])

    num = pd.DataFrame(index=raw.index, columns=cols, dtype=float)
    disp = pd.DataFrame(index=raw.index, columns=cols, dtype=object)
    for s, col in zip(spec, cols):
        key, fmt = s[2], s[3]
        step_key = s[5] if len(s) > 5 else None
        vals = pd.to_numeric(raw[key], errors='coerce') if key in raw else \
            pd.Series(float('nan'), index=raw.index)
        steps = raw[step_key] if step_key in raw else None
        sds = raw[key + '_std'] if key + '_std' in raw else None
        num[col] = vals
        disp[col] = [_cell(v, fmt,
                           None if steps is None else steps.iloc[i],
                           None if sds is None else sds.iloc[i])
                     for i, v in enumerate(vals)]

    sty = disp.style
    for s, col in zip(spec, cols):
        better = s[4]
        v = num[col]
        if better is None or v.notna().sum() < 2 or v.max() == v.min():
            continue
        # cap the darkest shade so text stays readable
        sty = sty.background_gradient(
            cmap=cmap if better == 'high' else f'{cmap}_r',
            subset=[col], gmap=v,
            low=0 if better == 'high' else strength,
            high=strength if better == 'high' else 0,
        )
        best = v.max() if better == 'high' else v.min()
        sty = sty.apply(lambda _c, v=v, best=best:
                        ['font-weight:700' if x == best else '' for x in v],
                        subset=[col])

    sty = sty.set_table_styles([
        {'selector': '',
         'props': 'border-collapse:collapse; font-family:DejaVu Sans,Helvetica,Arial,sans-serif;'
                  'font-size:12px; font-variant-numeric:tabular-nums;'},
        {'selector': 'caption',
         'props': 'caption-side:top; text-align:left; font-weight:600; padding:0 0 6px 0;'},
        {'selector': 'th',
         'props': 'padding:4px 8px; text-align:center; font-weight:600; color:#333;'},
        {'selector': 'th.col_heading.level0',
         'props': 'border-bottom:1px solid #999;'},
        {'selector': 'th.col_heading.level1',
         'props': 'border-bottom:1.5px solid #333; font-weight:500; white-space:nowrap;'},
        {'selector': 'th.row_heading',
         'props': 'text-align:left; white-space:nowrap; padding-right:14px;'},
        {'selector': 'td',
         'props': 'padding:4px 8px; text-align:right; white-space:nowrap;'},
        {'selector': 'tbody tr',
         'props': 'border-bottom:1px solid #eee;'},
    ])
    if caption:
        sty = sty.set_caption(caption)
    return sty


def _show(sty):
    """Display a Styler in a notebook; fall back to plain text elsewhere."""
    try:
        from IPython.display import display
        display(sty)
    except ImportError:
        print(sty.data.to_string())


def _load(path):
    return (pd.read_csv(path)
              .drop_duplicates(subset='step', keep='last')
              .sort_values('step'))


# ---------------------------------------------------------------------------
# Seeds and recompute
# ---------------------------------------------------------------------------
_SEED_RE = re.compile(r'(traj_xl\d+_xs\d+_s)(\d+)')
_RUN_RE = re.compile(
    r'^(?P<base>.*)/traj_xl(?P<xl>\d+)_xs(?P<xs>\d+)_s(?P<seed>\d+)/(?P<site>[^/]+)/'
    r'(?P<objective>[^_/]+)_(?P<arch>[^/]+)_(?P<scaling>scaled|unscaled)'
    r'(?:/alpha_(?P<alpha>[^/]+))?/trajectory\.csv$')


def _trained(csv_path):
    """A seed run counts as trained if its csv exists or its traj_probes/ dir does
    (i.e. training finished but trajectory.csv was never written)."""
    return (os.path.exists(csv_path) or
            os.path.isdir(os.path.join(os.path.dirname(csv_path), 'traj_probes')))


def seed_paths(path):
    """{seed: trajectory.csv path} for every sibling traj_xl.._xs.._s<seed> dir with the
    same sub-path that is trained (the csv itself may still be missing).
    Paths without a seed component map to {None: path}."""
    m = _SEED_RE.search(path)
    if not m:
        return {None: path}
    parent, prefix, rest = path[:m.start()], m.group(1), path[m.end():]
    found = {int(m.group(2)): path}
    if os.path.isdir(parent or '.'):
        for entry in os.listdir(parent or '.'):
            mm = re.fullmatch(re.escape(prefix) + r'(\d+)', entry)
            if mm and _trained(parent + entry + rest):
                found[int(mm.group(1))] = parent + entry + rest
    return dict(sorted(found.items()))


def parse_run_path(path):
    """Recover recompute_trajectory_metrics kwargs from a trajectory.csv path.
    Assumes the objective name has no underscore (e.g. 'mse', 'cos')."""
    m = _RUN_RE.match(path.replace(os.sep, '/'))
    if not m:
        raise ValueError(f"unrecognised run layout: {path}")
    g = m.groupdict()
    alpha = float(g['alpha'].replace('p', '.').replace('m', '-')) if g['alpha'] else 1.0
    return dict(base_align_head_dir=g['base'], traj_seed=int(g['seed']),
                traj_xl=int(g['xl']), traj_xs=int(g['xs']), site=g['site'],
                objective=g['objective'], arch=g['arch'],
                scaled=g['scaling'] == 'scaled', alpha=alpha)


_RECOMPUTED = set()   # csv paths already recomputed in this session


def _recompute_fn(recompute):
    """recompute['fn'] if given, else recompute_trajectory_metrics from the
    xl2xs_alignhead_traj module, else the one defined in the notebook."""
    if recompute.get('fn') is not None:
        return recompute['fn']
    try:
        from xl2xs_alignhead_traj import recompute_trajectory_metrics
        return recompute_trajectory_metrics
    except ImportError:
        import __main__
        fn = getattr(__main__, 'recompute_trajectory_metrics', None)
        if fn is None:
            raise ImportError("recompute_trajectory_metrics not found: import "
                              "xl2xs_alignhead_traj or pass recompute['fn']")
        return fn


def recompute_all(per, recompute, force=False):
    """Run recompute_trajectory_metrics once per seed csv before plotting.

    per: {name: {seed: csv_path}}
    recompute: dict with model_bank_dir, eval_batches, train_config, traj_config
               (traj_xs/traj_xl are taken from the path). Optional 'fn'.
    Always called with keyword args, so argument order can't go wrong.
    """
    fn = _recompute_fn(recompute)
    for by_seed in per.values():
        for path in by_seed.values():
            if path in _RECOMPUTED and not force:
                continue
            kw = parse_run_path(path)
            traj_config = {**recompute['traj_config'],
                           'traj_xl': kw.pop('traj_xl'), 'traj_xs': kw.pop('traj_xs')}
            try:
                fn(traj_config=traj_config, model_bank_dir=recompute['model_bank_dir'],
                   eval_batches=recompute['eval_batches'],
                   train_config=recompute['train_config'], write_csv=True, **kw)
                _RECOMPUTED.add(path)
            except Exception as e:          # keep going with whatever csv exists
                print(f"recompute failed for {path}: {type(e).__name__}: {e}")


def common_seeds(paths, recompute=None, force_recompute=False, verbose=True):
    """Seeds whose trajectory.csv exists for *every* path (after optional recompute).
    Returns (seeds, {name: {seed: csv}}); seeds == [None] means single-run fallback."""
    per = {name: seed_paths(p) for name, p in paths.items()}
    if recompute:
        recompute_all(per, recompute, force=force_recompute)

    have = {name: {s for s, p in d.items() if os.path.exists(p)} for name, d in per.items()}
    if verbose:
        for name, d in per.items():
            status = ', '.join(f"s{s}" + ('' if s in have[name] else ' (no trajectory.csv)')
                               for s in d if s is not None) or 'no seed in path'
            print(f"[seeds] {name}: {status}")

    common = set.intersection(*have.values()) if have else set()
    common.discard(None)
    if len(common) < 2:
        if verbose:
            missing = sorted({s for name, d in per.items() for s in d
                              if s is not None and s not in have[name]})
            hint = (f" Seeds {', '.join(f's{s}' for s in missing)} are trained but have no "
                    f"trajectory.csv — pass recompute=... to write them." if missing else "")
            print(f"[seeds] fewer than 2 seeds common to all paths; table uses the given "
                  f"paths only.{hint}")
        return [None], {name: {None: p} for name, p in paths.items()}
    seeds = sorted(common)
    if verbose:
        print(f"[seeds] table uses {', '.join(f's{s}' for s in seeds)}")
    return seeds, {name: {s: per[name][s] for s in seeds} for name in per}


_NAN = float('nan')


def _col(d, c):
    return d[c] if c in d and not d[c].isna().all() else None


def trajectory_metrics(d):
    """All summary numbers for one trajectory DataFrame (one seed, one arm)."""
    m = {}
    last = d.iloc[-1]
    r = _col(d, 'retained')
    r_pk = d.loc[r.idxmax()] if r is not None else None

    for pre, c in (('r', 'retained'), ('f', 'frozen_retained'), ('s', 'stitched_retained')):
        s = _col(d, c)
        if s is None:
            continue
        pk = d.loc[s.idxmax()]
        m.update({f'{pre}_peak': pk[c], f'{pre}_peak_step': pk.step,
                  f'{pre}_end': last[c], f'{pre}_drop': pk[c] - last[c]})

    for key, c in (('sim', 'sim_above_shuffle'), ('centered', 'centered_sim')):
        if _col(d, c) is not None:
            m[f'{key}_end'] = last[c]
            if r_pk is not None:
                m[key] = r_pk[c]

    if _col(d, 'xl_cycle_fvu') is not None:
        xl_var = d.xl_identity_mse / d.xl_cycle_fvu
        m['xl_cycle_fvu'] = last.xl_cycle_fvu
        m['xl_var'] = xl_var.iloc[-1]
        if 'xs2xl_mse' in d:
            m['xs2xl_fvu'] = (d.xs2xl_mse / xl_var).iloc[-1]
    if _col(d, 'xl_var') is not None:
        m['xl_var'] = last.xl_var
    if _col(d, 'xs_var') is not None:
        m['xs_var'] = last.xs_var
    return m


def _aggregate(per_seed_metrics, spec):
    """Mean over seeds (+ *_std when >1 seed); peak steps averaged and rounded."""
    keys = [s[2] for s in spec] + [s[5] for s in spec if len(s) > 5]
    recs = pd.DataFrame(per_seed_metrics).reindex(columns=keys)
    row = recs.mean().to_dict()
    if len(recs) > 1:
        row.update({f'{k}_std': v for k, v in recs.std(ddof=1).to_dict().items()})
        for s in spec:
            if len(s) > 5 and not _isnan(row[s[5]]):
                row[s[5]] = round(row[s[5]])
    return row


def seed_summary_table(paths, recompute=None, force_recompute=False, title=None,
                       spec=SUMMARY_SPEC, show=True):
    """One combined, shaded table over every seed available for *all* paths.

    paths     : {label: trajectory.csv of any one seed} — the same dict you pass to plots.
    recompute : dict(model_bank_dir, eval_batches, train_config, traj_config[, fn]).
                Seeds whose run dir exists without trajectory.csv are recomputed;
                each csv is recomputed at most once per session (force_recompute=True
                to redo). Omit to use the csvs as they are.
    Each csv is read once. Returns (numeric DataFrame, per-seed long DataFrame).
    """
    seeds, by_seed = common_seeds(paths, recompute, force_recompute)

    long_rows, rows = [], []
    for name, per in by_seed.items():
        per_seed = []
        for seed, p in per.items():
            m = trajectory_metrics(_load(p))
            per_seed.append(m)
            long_rows.append({'name': name, 'seed': seed, **m})
        rows.append({'name': name, **_aggregate(per_seed, spec)})

    df = pd.DataFrame(rows).set_index('name')
    keep = [s for s in spec if s[2] in df and df[s[2]].notna().any()]
    caption = title or 'Summary'
    if seeds != [None]:
        caption += f" — mean ± std over seeds {', '.join(f's{s}' for s in seeds)}"
    table = summary_table(rows, keep, caption=caption)
    if show:
        _show(table)
    return df, pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# Plots (single run per label, exactly the paths given)
# ---------------------------------------------------------------------------
def plot_functional_transfer_traj(paths, title=None, figsize=(13.5, 4), markers=True):
    """paths: {label: path_to_trajectory_csv}. Plots only; use seed_summary_table for numbers."""
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax = ax.ravel()
        mk = dict(marker='o', ms=2.5) if markers else {}

        for i, (name, path) in enumerate(paths.items()):
            d = (pd.read_csv(path)
                   .drop_duplicates(subset='step', keep='last')
                   .sort_values('step'))
            c = PALETTE[i % len(PALETTE)]

            fit = d[d.step > 0]
            ax[0].plot(d.step, d.retained * 100, color=c, label=name, **mk)
            ax[1].plot(fit.step, fit.frozen_retained * 100, color=c, label=name, **mk)
            ax[2].plot(fit.step, fit.stitched_retained * 100, color=c, label=name, **mk)

            # --- SAFETY CHECKS ---
            if d.retained.isna().all() or d.frozen_retained.isna().all():
                print(f"Skipping peak plotting for {name}: column contains only NaN")
                continue # Skip to next trajectory

            retrained_pk = d.loc[d.retained.idxmax()]
            frozen_pk = d.loc[d.frozen_retained.idxmax()]
            ax[0].plot(retrained_pk.step, retrained_pk.retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[1].plot(frozen_pk.step, frozen_pk.frozen_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[2].plot(frozen_pk.step, frozen_pk.stitched_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)

        ax[0].axhline(1.0, ls=(0, (4, 3)), c='#999999', lw=0.8, zorder=0)
        ax[0].set_ylabel('retained (%)')
        ax[0].set_title('Functional transfer - retrained probe on XL2XS', loc='left')
        ax[1].axhline(0, ls=(0, (4, 3)), c='#999999', lw=0.8, zorder=0)
        ax[1].set_ylabel('frozen retained (%)')
        ax[1].set_title('Functional transfer — XL2XS via frozen XS head', loc='left')
        ax[2].set_ylabel('stitched retained (%)')
        ax[2].set_title('Functional transfer — stitched downstream CE', loc='left')

        for a in ax:
            a.set_xlabel('align-head fit step')
            a.margins(x=0.02)
            a.legend(loc='lower right')

        if title:
            fig.suptitle(title, y=1.04, fontsize=11, x=0.005, ha='left')
        fig.tight_layout(w_pad=2.0)
        plt.show()


def plot_geometric_similarity_traj(paths, title=None, figsize=(8, 4), markers=True):
    """paths: {label: path_to_trajectory_csv}. Plots only; use seed_summary_table for numbers."""
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax = ax.ravel()
        mk = dict(marker='o', ms=2.5) if markers else {}

        for i, (name, path) in enumerate(paths.items()):
            d = (pd.read_csv(path)
                   .drop_duplicates(subset='step', keep='last')
                   .sort_values('step'))
            c = PALETTE[i % len(PALETTE)]

            fit = d[d.step > 0]
            ax[0].plot(d.step, d.sim_above_shuffle * 100, color=c, label=name, **mk)
            ax[1].plot(fit.step, fit.centered_sim * 100, color=c, label=name, **mk)

        ax[0].axhline(1.0, ls=(0, (4, 3)), c='#999999', lw=0.8, zorder=0)
        ax[0].set_ylabel('sim above shuffle')
        ax[0].set_title('Cosine sim above shuffle', loc='left')
        ax[1].axhline(0, ls=(0, (4, 3)), c='#999999', lw=0.8, zorder=0)
        ax[1].set_ylabel('centered similarity')
        ax[1].set_title('Centered similarity', loc='left')

        for a in ax:
            a.set_xlabel('align-head fit step')
            a.margins(x=0.02)
            a.legend(loc='lower right')

        if title:
            fig.suptitle(title, y=1.04, fontsize=11, x=0.005, ha='left')
        fig.tight_layout(w_pad=2.0)
        plt.show()


def plot_bidirectional(paths, title=None, figsize=(9, 4)):
    """Diagnose the bidirectional arm across alpha.

    paths: {label: trajectory.csv}  — one entry per alpha.
    """
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax = ax.ravel()

        for i, (name, path) in enumerate(paths.items()):
            d = (pd.read_csv(path)
                   .drop_duplicates(subset='step', keep='last')
                   .sort_values('step'))
            c = PALETTE[i % len(PALETTE)]

            xl_var = d.xl_identity_mse / d.xl_cycle_fvu
            xs2xl_fvu = d.xs2xl_mse / xl_var

            # --- the two halves of the trade ---
            ax[0].plot(d.step, d.xl_cycle_fvu, color=c, label=name)
            ax[0].plot(d.step, xs2xl_fvu, color=c, ls=':', alpha=.7)

            # --- convergence scatter ---
            ax[1].scatter(d.xl_cycle_fvu.iloc[-1], d.retained.iloc[-1] * 100,
                          color=c, s=70, zorder=5, label=name)
            ax[1].annotate(name.split()[-1], (d.xl_cycle_fvu.iloc[-1],
                                              d.retained.iloc[-1] * 100),
                           textcoords='offset points', xytext=(7, -3), fontsize=8, color=c)

        ax[0].set_ylabel('FVU')
        ax[0].set_xlabel('align-head fit step')
        ax[0].set_title('XL preserved  (solid: cycle, dotted: up)', loc='left')
        ax[1].set_xlabel('xl_cycle_fvu at convergence')
        ax[1].set_ylabel('retained (%)')
        ax[1].set_title('Preservation vs transfer', loc='left')

        for a in ax:
            a.margins(x=0.02)
            a.legend(loc='upper right')

        if title:
            fig.suptitle(title, y=1.02, fontsize=11, x=0.005, ha='left')
        fig.tight_layout(w_pad=2.0)
        plt.show()
