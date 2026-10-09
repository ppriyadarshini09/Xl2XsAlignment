# @title Eval plot libraries
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


def plot_functional_transfer_traj(paths, title=None, figsize=(13.5, 4), markers=True):
    """paths: {label: path_to_trajectory_csv}"""
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        ax = ax.ravel()
        summary = []
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
                # Append to summary with NaNs or dummy values to prevent unpacking errors later
                summary.append((name, float('nan'), float('nan'), -1, float('nan'),
                                float('nan'), float('nan'), float('nan'), -1,
                                float('nan'), float('nan'), float('nan')))
                continue # Skip to next trajectory

            retrained_pk = d.loc[d.retained.idxmax()]
            retrained_end = d.retained.iloc[-1]
            frozen_pk = d.loc[d.frozen_retained.idxmax()]
            frozen_end = d.frozen_retained.iloc[-1]
            stitched_pk = d.loc[d.stitched_retained.idxmax()]
            stitched_end = d.stitched_retained.iloc[-1]
            ax[0].plot(retrained_pk.step, retrained_pk.retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[1].plot(frozen_pk.step, frozen_pk.frozen_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)
            ax[2].plot(frozen_pk.step, frozen_pk.stitched_retained * 100, marker='o', ms=6, mfc='none',
                       mec=c, mew=1.4, ls='none', zorder=5)

            summary.append((name,
                            retrained_pk.gap_nats,
                            retrained_pk.retained,
                            int(retrained_pk.step),
                            retrained_end,
                            retrained_pk.retained - retrained_end,
                            retrained_pk.sim_above_shuffle,
                            d.sim_above_shuffle.iloc[-1],
                            d.xs_var.iloc[-1],
                            d.xl_var.iloc[-1],
                            frozen_pk.frozen_retained,
                            int(frozen_pk.step),
                            frozen_end,
                            frozen_pk.sim_above_shuffle,
                            frozen_pk.frozen_retained - frozen_end,
                            stitched_pk.stitched_retained,
                            int(stitched_pk.step),
                            stitched_end,
                            stitched_pk.sim_above_shuffle,
                            stitched_pk.stitched_retained - stitched_end,
                            ))

        for (name, gap_nats, r_peak, r_pstep, r_end, r_drop, r_peak_sim, r_sim_end,
             xs_var, xl_var,
             f_peak, f_pstep, f_end, f_peak_sim, f_drop,
             s_peak, s_pstep, s_end, s_peak_sim, s_drop) in summary:
            print(f"{name:15s} | gap nats {gap_nats:3f} "
                  f"| retrained peak {r_peak:3.1%} (step {r_pstep:3d})->{r_end:3.1%} "
                  f"| sim above shuffle {r_peak_sim:3.1%} (step {r_pstep:3d})->{r_sim_end:3.1%} "
                  f"| frozen peak {f_peak:3.1%} (step {f_pstep:3d})->{f_end:3.1%}"
                  f"| stitched peak {s_peak:3.1%} (step {s_pstep:3d})->{s_end:3.1%}"
                  f"| XL var {xl_var:3.1} | XS var {xs_var:3.1}")


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
    return summary


def plot_geometric_similarity_traj(paths, title=None, figsize=(8, 4), markers=True):
    """paths: {label: path_to_trajectory_csv}"""
    with mpl.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax = ax.ravel()
        summary = []
        mk = dict(marker='o', ms=2.5) if markers else {}

        for i, (name, path) in enumerate(paths.items()):
            d = (pd.read_csv(path)
                   .drop_duplicates(subset='step', keep='last')
                   .sort_values('step'))
            c = PALETTE[i % len(PALETTE)]

            fit = d[d.step > 0]
            ax[0].plot(d.step, d.sim_above_shuffle * 100, color=c, label=name, **mk)
            ax[1].plot(fit.step, fit.centered_sim * 100, color=c, label=name, **mk)

            # --- SAFETY CHECKS ---
            if d.retained.isna().all() or d.frozen_retained.isna().all():
                print(f"Skipping peak plotting for {name}: column contains only NaN")
                # Append to summary with NaNs or dummy values to prevent unpacking errors later
                summary.append((name, float('nan'), float('nan'), -1, float('nan'),
                                float('nan'), float('nan'), float('nan'), -1,
                                float('nan'), float('nan'), float('nan')))
                continue # Skip to next trajectory

            retrained_pk = d.loc[d.retained.idxmax()]
            retrained_end = d.retained.iloc[-1]
            frozen_end = d.frozen_retained.iloc[-1]
            stitched_end = d.stitched_retained.iloc[-1]

            summary.append((name,
                            retrained_pk.gap_nats,
                            retrained_pk.retained,
                            int(retrained_pk.step),
                            retrained_end,
                            retrained_pk.sim_above_shuffle,
                            retrained_pk.retained - retrained_end,
                            retrained_pk.frozen_retained,
                            retrained_pk.stitched_retained,
                            retrained_pk.sim_above_shuffle,
                            retrained_pk.centered_sim,
                            retrained_pk.xs_scale,
                            retrained_pk.xl_scale,
                            ))

        for (name, gap_nats, r_peak, r_pstep, r_end, r_peak_sim, r_drop,
             r_frozen, r_stitched, r_sim, r_cen, xs_scale, xl_scale) in summary:
            print(f"{name:15s} | gap nats {gap_nats:3f} "
                  f"| retrained peak {r_peak:3.1%} (step {r_pstep:3d})->{r_end:3.1%} | "
                  f" @ retrained peak (frozen:  {r_frozen:3.1%}, "
                  f"stitched: {r_stitched:3.1%}, sim above shuffle: "
                  f"{r_sim:3.1%}, centered sim: {r_cen:3.1%} | "
                  f"XS scale {xs_scale:6.1}, XL scale {xl_scale:6.1}")

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
    return summary


def plot_bidirectional(paths, title=None, figsize=(9, 4)):
    """Diagnose the bidirectional arm across alpha.

    paths: {label: trajectory.csv}  — one entry per alpha.
    """
    summary = []
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
            xl_cycle_fvu_end = d.xl_cycle_fvu.iloc[-1]
            xs2xl_fvu_end = xs2xl_fvu.iloc[-1]
            summary.append((name,
                            d.xl_cycle_fvu.iloc[-1],
                            xs2xl_fvu.iloc[-1],
                            xl_var.iloc[-1]))

        for (name, cycle_fvu, xs2xl_fvu, xl_var) in summary:
          print(f"{name:15s} | xl_cycle_fvu {cycle_fvu:3f} "
                f"| xs2xl_fvu {xs2xl_fvu:3f} | xl var {xl_var:3f}")

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
        return summary
