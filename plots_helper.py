# @title Eval plot libraries
import math

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


def _cell(v, fmt, step=None):
    """Format one cell; NaN -> em dash, optional ' @ step' suffix."""
    if _isnan(v):
        return '—'
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
        num[col] = vals
        disp[col] = [_cell(v, fmt, None if steps is None else steps.iloc[i])
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


def _empty_row(name, spec):
    return {'name': name, **{s[2]: float('nan') for s in spec}}  # steps -> NaN via reindex


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_functional_transfer_traj(paths, title=None, figsize=(13.5, 4), markers=True,
                                  show_table=True):
    """paths: {label: path_to_trajectory_csv}. Returns the summary DataFrame."""
    rows = []
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
                rows.append(_empty_row(name, FUNCTIONAL_SPEC))
                continue

            r_pk = d.loc[d.retained.idxmax()]
            f_pk = d.loc[d.frozen_retained.idxmax()]
            s_pk = d.loc[d.stitched_retained.idxmax()]
            last = d.iloc[-1]

            ax[0].plot(r_pk.step, r_pk.retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[1].plot(f_pk.step, f_pk.frozen_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[2].plot(f_pk.step, f_pk.stitched_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)

            rows.append(dict(
                name=name,
                gap_nats=r_pk.gap_nats,
                r_peak=r_pk.retained, r_peak_step=r_pk.step,
                r_end=last.retained, r_drop=r_pk.retained - last.retained,
                r_peak_sim=r_pk.sim_above_shuffle, sim_end=last.sim_above_shuffle,
                f_peak=f_pk.frozen_retained, f_peak_step=f_pk.step,
                f_end=last.frozen_retained, f_drop=f_pk.frozen_retained - last.frozen_retained,
                s_peak=s_pk.stitched_retained, s_peak_step=s_pk.step,
                s_end=last.stitched_retained, s_drop=s_pk.stitched_retained - last.stitched_retained,
                xs_var=last.xs_var, xl_var=last.xl_var,
            ))

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

    table = summary_table(rows, FUNCTIONAL_SPEC, caption=title or 'Functional transfer')
    if show_table:
        _show(table)
    return pd.DataFrame(rows).set_index('name')


def plot_geometric_similarity_traj(paths, title=None, figsize=(8, 4), markers=True,
                                   show_table=True):
    """paths: {label: path_to_trajectory_csv}. Returns the summary DataFrame."""
    rows = []
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
                rows.append(_empty_row(name, GEOMETRIC_SPEC))
                continue

            pk = d.loc[d.retained.idxmax()]
            end = d.retained.iloc[-1]
            rows.append(dict(
                name=name,
                gap_nats=pk.gap_nats,
                r_peak=pk.retained, r_peak_step=pk.step,
                r_end=end, r_drop=pk.retained - end,
                frozen=pk.frozen_retained, stitched=pk.stitched_retained,
                sim=pk.sim_above_shuffle, centered=pk.centered_sim,
                sim_end=d.sim_above_shuffle.iloc[-1], centered_end=d.centered_sim.iloc[-1],
                xs_scale=pk.xs_scale, xl_scale=pk.xl_scale,
            ))

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

    table = summary_table(rows, GEOMETRIC_SPEC, caption=title or 'Geometric similarity')
    if show_table:
        _show(table)
    return pd.DataFrame(rows).set_index('name')


def plot_bidirectional(paths, title=None, figsize=(9, 4), show_table=True):
    """Diagnose the bidirectional arm across alpha.

    paths: {label: trajectory.csv} — one entry per alpha. Returns the summary DataFrame.
    """
    rows = []
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

            xs_var = d.xs_var.iloc[-1] if 'xs_var' in d else float('nan')
            rows.append(dict(name=name, xl_cycle_fvu=x_end, xs2xl_fvu=xs2xl_fvu.iloc[-1],
                             xl_var=xl_var.iloc[-1], xs_var=xs_var, retained=r_end))

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

    table = summary_table(rows, BIDIRECTIONAL_SPEC, caption=title or 'Bidirectional arm')
    if show_table:
        _show(table)
    return pd.DataFrame(rows).set_index('name')
