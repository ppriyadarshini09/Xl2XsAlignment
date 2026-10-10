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
# Summary tables
# ---------------------------------------------------------------------------
# Each table is described by a spec: an ordered list of
#   (group, label, key, fmt, better[, step_key])
# group    : top-level header (columns with the same group are merged)
# label    : sub-header shown under the group
# key      : key in the per-run summary dict
# fmt      : format string for the cell
# better   : 'high' / 'low' -> shaded green where favourable; None -> no shading
# step_key : optional; appends " @ <step>" to the cell (shading still uses `key`)
PCT, FLT, INT, SCI = '{:.1%}', '{:.3f}', '{:d}', '{:.2e}'

FUNCTIONAL_SPEC = [
    ('Retrained', 'peak @ step', 'r_peak', PCT, 'high', 'r_peak_step'),
    ('Retrained', 'end',         'r_end',  PCT, 'high'),
    ('Retrained', 'drop',        'r_drop', PCT, 'low'),
    ('Frozen',    'peak @ step', 'f_peak', PCT, 'high', 'f_peak_step'),
    ('Frozen',    'end',         'f_end',  PCT, 'high'),
    ('Frozen',    'drop',        'f_drop', PCT, 'low'),
    ('Stitched',  'peak @ step', 's_peak', PCT, 'high', 's_peak_step'),
    ('Stitched',  'end',         's_end',  PCT, 'high'),
    ('Stitched',  'drop',        's_drop', PCT, 'low'),
]

GEOMETRIC_SPEC = [
    ('Sim > shuffle', '@ retrained peak', 'sim',          PCT, 'high', 'r_peak_step'),
    ('Sim > shuffle', 'end',              'sim_end',      PCT, 'high'),
    ('Centered sim',  '@ retrained peak', 'centered',     PCT, 'high', 'r_peak_step'),
    ('Centered sim',  'end',              'centered_end', PCT, 'high'),
]

BIDIRECTIONAL_SPEC = [
    ('At convergence', 'XL cycle FVU', 'xl_cycle_fvu', FLT, 'low'),
    ('At convergence', 'XS→XL FVU',    'xs2xl_fvu',    FLT, 'low'),
    ('At convergence', 'XL var',       'xl_var',       SCI, None),
    ('At convergence', 'XS var',       'xs_var',       SCI, None),
    ('At convergence', 'retained',     'retained',     PCT, 'high'),
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


def seed_paths(path):
    """{seed: trajectory.csv} for every sibling traj_xl.._xs.._s<seed> dir that has
    the same sub-path. Paths without a seed component map to {None: path}."""
    m = _SEED_RE.search(path)
    if not m:
        return {None: path}
    parent, prefix, rest = path[:m.start()], m.group(1), path[m.end():]
    found = {int(m.group(2)): path}
    if os.path.isdir(parent or '.'):
        for entry in os.listdir(parent or '.'):
            mm = re.fullmatch(re.escape(prefix) + r'(\d+)', entry)
            if mm and os.path.exists(parent + entry + rest):
                found[int(mm.group(1))] = parent + entry + rest
    return dict(sorted(found.items()))


def common_seeds(paths):
    """Seeds present for *every* path. Falls back to the given paths only (seed=None)."""
    per = {name: seed_paths(p) for name, p in paths.items()}
    common = set.intersection(*(set(s) for s in per.values())) if per else set()
    common.discard(None)
    if len(common) < 2:
        return [None], {name: {None: p} for name, p in paths.items()}
    seeds = sorted(common)
    return seeds, {name: {s: per[name][s] for s in seeds} for name in per}


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


def recompute_all(paths_by_seed, recompute, force=False):
    """Run recompute_trajectory_metrics once per csv (all seeds) before plotting.

    recompute: dict with model_bank_dir, eval_batches, train_config, traj_config
               (traj_xs/traj_xl are taken from the path). Optional 'fn' to override
               the recompute function (default: xl2xs_alignhead_traj's).
    """
    fn = recompute.get('fn')
    if fn is None:
        from xl2xs_alignhead_traj import recompute_trajectory_metrics as fn
    for by_seed in paths_by_seed.values():
        for path in by_seed.values():
            if path in _RECOMPUTED and not force:
                continue
            a = parse_run_path(path)
            traj_config = {**recompute['traj_config'],
                           'traj_xl': a.pop('traj_xl'), 'traj_xs': a.pop('traj_xs')}
            try:
                fn(traj_config=traj_config, model_bank_dir=recompute['model_bank_dir'],
                   eval_batches=recompute['eval_batches'],
                   train_config=recompute['train_config'], write_csv=True, **a)
                _RECOMPUTED.add(path)
            except Exception as e:          # keep plotting with the existing csv
                print(f"recompute failed for {path}: {e}")


def _prepare(paths, recompute, force_recompute):
    seeds, by_seed = common_seeds(paths)
    if recompute:
        recompute_all(by_seed, recompute, force=force_recompute)
    return seeds, by_seed


def seed_rows(by_seed, row_fn, spec):
    """One row per run name: the single-seed row, or mean (+ *_std) across seeds."""
    keys = [s[2] for s in spec] + [s[5] for s in spec if len(s) > 5]
    rows = []
    for name, per_seed in by_seed.items():
        recs = pd.DataFrame([row_fn(_load(p)) for p in per_seed.values()]).reindex(columns=keys)
        row = {'name': name, **recs.mean().to_dict()}
        if len(recs) > 1:
            row.update({f'{k}_std': v for k, v in recs.std(ddof=1).to_dict().items()})
            for s in spec:                      # steps: mean, rounded to an int
                if len(s) > 5 and not _isnan(row[s[5]]):
                    row[s[5]] = round(row[s[5]])
        rows.append(row)
    return rows


def _caption(base, seeds):
    if seeds == [None]:
        return base
    return f"{base} — mean ± std over seeds {', '.join(f's{s}' for s in seeds)}"


_NAN = float('nan')


def functional_row(d):
    if any(d[c].isna().all() for c in ('retained', 'frozen_retained', 'stitched_retained')):
        return {}
    r_pk = d.loc[d.retained.idxmax()]
    f_pk = d.loc[d.frozen_retained.idxmax()]
    s_pk = d.loc[d.stitched_retained.idxmax()]
    last = d.iloc[-1]
    return dict(
        r_peak=r_pk.retained, r_peak_step=r_pk.step,
        r_end=last.retained, r_drop=r_pk.retained - last.retained,
        f_peak=f_pk.frozen_retained, f_peak_step=f_pk.step,
        f_end=last.frozen_retained, f_drop=f_pk.frozen_retained - last.frozen_retained,
        s_peak=s_pk.stitched_retained, s_peak_step=s_pk.step,
        s_end=last.stitched_retained, s_drop=s_pk.stitched_retained - last.stitched_retained,
    )


def geometric_row(d):
    if d.retained.isna().all():
        return {}
    pk = d.loc[d.retained.idxmax()]
    return dict(r_peak_step=pk.step,
                sim=pk.sim_above_shuffle, sim_end=d.sim_above_shuffle.iloc[-1],
                centered=pk.centered_sim, centered_end=d.centered_sim.iloc[-1])


def bidirectional_row(d):
    xl_var = d.xl_identity_mse / d.xl_cycle_fvu
    return dict(xl_cycle_fvu=d.xl_cycle_fvu.iloc[-1],
                xs2xl_fvu=(d.xs2xl_mse / xl_var).iloc[-1],
                xl_var=xl_var.iloc[-1],
                xs_var=d.xs_var.iloc[-1] if 'xs_var' in d else _NAN,
                retained=d.retained.iloc[-1])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_functional_transfer_traj(paths, title=None, figsize=(13.5, 4), markers=True,
                                  show_table=True, recompute=None, force_recompute=False):
    """paths: {label: path_to_trajectory_csv}. Plots use these paths as given; the table
    uses every seed that exists for *all* paths (mean ± std). Pass `recompute` (see
    recompute_all) to refresh each csv once before plotting. Returns the summary DataFrame."""
    seeds, by_seed = _prepare(paths, recompute, force_recompute)
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax = ax.ravel()
        mk = dict(marker='o', ms=2.5) if markers else {}

        for i, (name, path) in enumerate(paths.items()):
            d = _load(path)
            c = PALETTE[i % len(PALETTE)]

            fit = d[d.step > 0]
            ax[0].plot(d.step, d.retained * 100, color=c, label=name, **mk)
            ax[1].plot(fit.step, fit.frozen_retained * 100, color=c, label=name, **mk)
            ax[2].plot(fit.step, fit.stitched_retained * 100, color=c, label=name, **mk)

            needed = ['retained', 'frozen_retained', 'stitched_retained']
            if any(d[col].isna().all() for col in needed):
                print(f"Skipping peaks for {name}: a retained column is all NaN")
                continue

            r_pk = d.loc[d.retained.idxmax()]
            f_pk = d.loc[d.frozen_retained.idxmax()]
            s_pk = d.loc[d.stitched_retained.idxmax()]

            ax[0].plot(r_pk.step, r_pk.retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[1].plot(f_pk.step, f_pk.frozen_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[2].plot(f_pk.step, f_pk.stitched_retained * 100, marker='o', ms=6, mfc='none',
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

    rows = seed_rows(by_seed, functional_row, FUNCTIONAL_SPEC)
    table = summary_table(rows, FUNCTIONAL_SPEC, caption=_caption(title or 'Functional transfer', seeds))
    if show_table:
        _show(table)
    return pd.DataFrame(rows).set_index('name')


def plot_geometric_similarity_traj(paths, title=None, figsize=(8, 4), markers=True,
                                   show_table=True, recompute=None, force_recompute=False):
    """paths: {label: path_to_trajectory_csv}. See plot_functional_transfer_traj for
    seeds / recompute. Returns the summary DataFrame."""
    seeds, by_seed = _prepare(paths, recompute, force_recompute)
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax = ax.ravel()
        mk = dict(marker='o', ms=2.5) if markers else {}

        for i, (name, path) in enumerate(paths.items()):
            d = _load(path)
            c = PALETTE[i % len(PALETTE)]

            fit = d[d.step > 0]
            ax[0].plot(d.step, d.sim_above_shuffle * 100, color=c, label=name, **mk)
            ax[1].plot(fit.step, fit.centered_sim * 100, color=c, label=name, **mk)

            if d.retained.isna().all():
                print(f"Skipping peaks for {name}: retained is all NaN")
                continue

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

    rows = seed_rows(by_seed, geometric_row, GEOMETRIC_SPEC)
    table = summary_table(rows, GEOMETRIC_SPEC, caption=_caption(title or 'Geometric similarity', seeds))
    if show_table:
        _show(table)
    return pd.DataFrame(rows).set_index('name')


def plot_bidirectional(paths, title=None, figsize=(9, 4), show_table=True,
                       recompute=None, force_recompute=False):
    """Diagnose the bidirectional arm across alpha.

    paths: {label: trajectory.csv} — one entry per alpha. See plot_functional_transfer_traj
    for seeds / recompute. Returns the summary DataFrame.
    """
    seeds, by_seed = _prepare(paths, recompute, force_recompute)
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax = ax.ravel()

        for i, (name, path) in enumerate(paths.items()):
            d = _load(path)
            c = PALETTE[i % len(PALETTE)]

            xl_var = d.xl_identity_mse / d.xl_cycle_fvu
            xs2xl_fvu = d.xs2xl_mse / xl_var

            ax[0].plot(d.step, d.xl_cycle_fvu, color=c, label=name)
            ax[0].plot(d.step, xs2xl_fvu, color=c, ls=':', alpha=.7)

            x_end, r_end = d.xl_cycle_fvu.iloc[-1], d.retained.iloc[-1]
            ax[1].scatter(x_end, r_end * 100, color=c, s=70, zorder=5, label=name)
            ax[1].annotate(name.split()[-1], (x_end, r_end * 100),
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

    rows = seed_rows(by_seed, bidirectional_row, BIDIRECTIONAL_SPEC)
    table = summary_table(rows, BIDIRECTIONAL_SPEC, caption=_caption(title or 'Bidirectional arm', seeds))
    if show_table:
        _show(table)
    return pd.DataFrame(rows).set_index('name')
