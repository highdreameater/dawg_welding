"""Export the actor of an Isaac Lab RSL-RL checkpoint (checkpoint.pt) to a TorchScript policy.pt.

The Isaac Lab locomotion actors are plain MLPs with ELU activations and no observation
normalisation, so the exported module maps obs -> action exactly like the Isaac Sim policies.

Usage:
    python tools/export_rslrl_checkpoint.py policies/anymal/Isaac-Velocity-Rough-Anymal-C-v0/checkpoint.pt
"""
import argparse
import os

import torch
import torch.nn as nn


def build_actor(state_dict):
    keys = sorted({int(k.split(".")[1]) for k in state_dict if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for i, idx in enumerate(keys):
        w = state_dict[f"actor.{idx}.weight"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(state_dict[f"actor.{idx}.bias"])
        layers.append(lin)
        if i < len(keys) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = build_actor(ck["model_state_dict"]).eval()
    print(actor)
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "policy.pt")
    torch.jit.script(actor).save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
