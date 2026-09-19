import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_gpt import get_batch, device


def train_standalone_probe(
    model,
    site,
    rs_scale,
    train_data,
    eval_batches,
    config,
    probe_dir,
    run_name,
    max_steps=2500,
    eval_every=250,
    lr=1e-3,
    weight_decay=0.01,
):
    model.eval()
    V = config["vocab_size"]
    n_embed = model.n_embed
    probe_path = os.path.join(probe_dir, f"probe_{n_embed}.pt")

    # Init probe head
    probe = nn.Linear(n_embed, V, bias=False).to(device)

    if os.path.exists(probe_path):
        probe_ckpt = torch.load(probe_path, map_location=device)
        if probe_ckpt["step"] == max_steps:
            probe.load_state_dict(probe_ckpt["probe"])
            print(
                f"Trained probe for run_name: '{run_name}' already available @ {probe_path}"
            )
            return probe, probe_ckpt["best_loss"]
        else:
            print(f"Starting probe training for run_name: {run_name}...")

    nn.init.normal_(probe.weight, mean=0.0, std=0.02)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")

    @torch.no_grad()
    def eval_probe():
        losses = []
        probe.eval()
        for x, y in eval_batches:
            logits = probe(model.forward_repr_at_site(site, x) / rs_scale)
            B, T, _ = logits.shape
            losses.append(F.cross_entropy(logits.view(B * T, V), y.view(B * T)).item())
        probe.train()
        return float(np.mean(losses))

    for step in range(max_steps + 1):
        x, y = get_batch(train_data, config["block_size"], config["batch_size"])
        with torch.no_grad():
            repr_ = model.forward_repr_at_site(site, x) / rs_scale
        logits = probe(repr_)
        B, T, _ = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % eval_every == 0:
            eval_loss = eval_probe()
            str_msg = f"step {step} | train: {loss.item():.4f}, eval: {eval_loss:.4f}"
            if best_loss > eval_loss:
                best_loss = eval_loss
                str_msg = str_msg + f" <- best"
            print(str_msg)

        if step >= max_steps:
            torch.save(
                {
                    "step": step,
                    "n_embed": n_embed,
                    "probe": probe.state_dict(),
                    "best_loss": best_loss,
                },
                probe_path,
            )

    print(f"Probe training complete; best loss -> {best_loss:.4f}")
    return probe, best_loss


def train_xl2xs_probe(
    xs_model,
    xl_model,
    site,
    xs_scale,
    xl_scale,
    head,
    train_data,
    eval_batches,
    config,
    traj_probe_dir,
    traj_step_pt,
    steps=3000,
    eval_every=250,
    lr=1e-3,
    weight_decay=0.01,
):
    xs_n_embed = xs_model.n_embed
    xl_n_embed = xl_model.n_embed
    V = config["vocab_size"]
    probe_path = os.path.join(traj_probe_dir, traj_step_pt)

    # init probe
    probe = nn.Linear(xs_n_embed, V, bias=False).to(device)

    if os.path.exists(probe_path):
        ckpt = torch.load(probe_path, map_location=device)
        if ckpt["step"] >= steps:
            probe.load_state_dict(ckpt["probe"])
            best_loss = ckpt["best_loss"]
            print(f"xl2xs probe already available @ {probe_path}")
            return probe, best_loss

    print(
        f"Starting xl({xl_n_embed})2xs({xs_n_embed}) for "
        f"{traj_step_pt} probe training..."
    )
    nn.init.normal_(probe.weight, mean=0.0, std=0.02)
    optim = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    xl_model.eval()
    head.eval()

    @torch.no_grad()
    def eval_probe():
        losses = []
        probe.eval()
        for x, y in eval_batches:
            B, T = y.shape
            xl = xl_model.forward_repr_at_site(site, x) / xl_scale
            xl2xs = head(xl)
            loss = F.cross_entropy(probe(xl2xs).view(B * T, V), y.view(B * T))
            losses.append(loss.item())
        probe.train()
        return sum(losses) / len(losses)

    best_loss = float("inf")
    for step in range(steps + 1):
        x, y = get_batch(train_data, config["block_size"], config["batch_size"])

        # forward
        with torch.no_grad():
            xl2xs = head(xl_model.forward_repr_at_site(site, x) / xl_scale)
        logits = probe(xl2xs)
        B, T, _ = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        # backward
        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % eval_every == 0:
            eval_loss = eval_probe()
            str_msg = f"step {step} | train: {loss.item():.4f}, eval: {eval_loss:.4f}"
            if best_loss > eval_loss:
                best_loss = eval_loss
                str_msg = str_msg + f" <- best"
            print(str_msg)

        if step >= steps:
            torch.save(
                {
                    "step": step,
                    "xs_n_embed": xs_n_embed,
                    "xl_n_embed": xl_n_embed,
                    "probe": probe.state_dict(),
                    "best_loss": best_loss,
                },
                probe_path,
            )

    print(
        f"Finished XL:{xl_n_embed} to XS:{xs_n_embed} probe training; "
        f"best loss {best_loss:.4f}"
    )
    return probe, best_loss
