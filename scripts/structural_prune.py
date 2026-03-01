#!/usr/bin/env python3
"""Structural pruning for Qwen3.5-27B → Q4_K_M pipeline.

Two-pass approach:
  Pass 1 (read-only): Compute equilibrium group scores for importance ranking
  Pass 2 (write): Physically remove tensor dimensions and write pruned shards

Uses game-theoretic equilibrium pruning (arXiv:2512.22106) for importance scoring.

Usage:
    python structural_prune.py --model-dir /tmp/qwen35-fp16 --dry-run
    python structural_prune.py --model-dir /tmp/qwen35-fp16 --output-dir /tmp/qwen35-pruned
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Protected layers: never remove first/last N
PROTECTED_FIRST = 4
PROTECTED_LAST = 4

# Tensor name patterns
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
SKIP_PATTERNS = ["embed_tokens", "lm_head", "norm", "layernorm"]


# ---------------------------------------------------------------------------
# Equilibrium group scoring
# ---------------------------------------------------------------------------

def equilibrium_group_scores(
    weight: torch.Tensor,
    group_size: int,
    steps: int = 20,
    lr: float = 0.1,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.05,
) -> torch.Tensor:
    """Compute group-level equilibrium participation scores.

    Each group of ``group_size`` rows is treated as a player in a game.
    The game balances importance (benefit) against cost and L1 pressure,
    producing sharper important/unimportant separation than raw L2 norms.

    Args:
        weight: Tensor of shape (out_features, in_features).
        group_size: Number of rows per group (head_dim, or 1 for channels).
        steps: Number of equilibrium iterations.
        lr: Learning rate for participation update.
        alpha: Benefit coefficient (importance weight).
        beta: Cost coefficient (redundancy penalty).
        gamma: L1 regularisation strength (sparsity pressure).

    Returns:
        Participation scores of shape ``(num_groups,)`` in [0, 1].
    """
    if weight.dim() < 2 or weight.shape[0] < group_size:
        return torch.ones(max(1, weight.shape[0] // max(group_size, 1)))

    w = weight.to(DEVICE).float()
    num_groups = w.shape[0] // group_size

    # Reshape to groups: (num_groups, group_size, in_features)
    w_groups = w[: num_groups * group_size].view(num_groups, group_size, -1)

    # Global normalisation (not per-group) so relative magnitudes are preserved
    global_max = w_groups.abs().max().clamp(min=1e-8)
    w_norm = w_groups / global_max

    # Group norm squared — serves as importance proxy
    norm_sq = (w_norm**2).sum(dim=(1, 2))  # (num_groups,)

    # Initialise participation from magnitude
    s = (norm_sq / norm_sq.max().clamp(min=1e-8)).clone()

    # Equilibrium game loop
    two_beta = 2.0 * beta
    for _ in range(steps):
        benefit = alpha * norm_sq
        cost = two_beta * norm_sq * s
        l1 = gamma * (s > 0).float()
        s = (s + lr * (benefit - cost - l1)).clamp(0, 1)

    return s.cpu()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_layer_idx(name: str) -> Optional[int]:
    """Extract layer index from a tensor name like ``model.layers.3.…``."""
    m = LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def is_full_attention(layer_idx: int, full_attn_interval: int) -> bool:
    """Return True if *layer_idx* is a full-attention layer."""
    if full_attn_interval <= 0:
        return False
    return (layer_idx + 1) % full_attn_interval == 0


def rename_layer(name: str, old_idx: int, new_idx: int) -> str:
    """Replace ``layers.{old_idx}.`` with ``layers.{new_idx}.`` in *name*."""
    return name.replace(f"layers.{old_idx}.", f"layers.{new_idx}.")


# ---------------------------------------------------------------------------
# Pass 1 — importance scoring
# ---------------------------------------------------------------------------

def score_pass(model_dir: Path, config: dict) -> dict:
    """Read all shards and compute per-group importance scores."""
    shards = sorted(model_dir.glob("*.safetensors"))
    num_layers = config.get("num_hidden_layers", 64)
    # Default to 1 (all layers are full attention) for standard transformers.
    # Qwen3.5 DeltaNet uses interval=4 (3 linear + 1 full).
    full_attn_interval = config.get("full_attention_interval", 1)

    # Accumulators — will be initialised on first encounter
    mlp_channel_scores: Optional[torch.Tensor] = None
    mlp_layer_count = 0

    full_head_scores: Optional[torch.Tensor] = None
    full_head_count = 0

    linear_v_scores: Optional[torch.Tensor] = None
    linear_v_count = 0

    layer_scores: dict[int, float] = {}
    layer_types: dict[int, str] = {}

    print(f"Pass 1: Scoring {len(shards)} shards…", flush=True)

    for shard_i, shard in enumerate(shards):
        print(f"\n  [{shard_i + 1}/{len(shards)}] {shard.name}", flush=True)

        with safe_open(str(shard), framework="pt", device="cpu") as f:
            # Group tensors by layer
            by_layer: dict[int, dict[str, torch.Tensor]] = {}
            for name in f.keys():
                idx = parse_layer_idx(name)
                if idx is not None:
                    by_layer.setdefault(idx, {})[name] = f.get_tensor(name)

            for idx in sorted(by_layer):
                tensors = by_layer[idx]
                ltype = (
                    "full_attention"
                    if is_full_attention(idx, full_attn_interval)
                    else "linear_attention"
                )
                layer_types[idx] = ltype
                group_means: list[float] = []

                for name, tensor in tensors.items():
                    if any(p in name for p in SKIP_PATTERNS):
                        continue
                    if tensor.dim() < 2:
                        continue

                    # --- MLP channels (all layers) ---
                    if "gate_proj" in name or "up_proj" in name:
                        scores = equilibrium_group_scores(tensor, group_size=1)
                        if mlp_channel_scores is None:
                            mlp_channel_scores = torch.zeros(scores.shape[0])
                        mlp_channel_scores += scores
                        mlp_layer_count += 1
                        group_means.append(scores.mean().item())
                        print(f"    {name}: MLP channels ({scores.shape[0]})", flush=True)

                    # --- Full-attention Q heads ---
                    elif "q_proj" in name and ltype == "full_attention":
                        n_heads = config.get("num_attention_heads", 24)
                        head_dim = tensor.shape[0] // n_heads
                        scores = equilibrium_group_scores(tensor, group_size=head_dim)
                        if full_head_scores is None:
                            full_head_scores = torch.zeros(n_heads)
                        full_head_scores += scores[:n_heads]
                        full_head_count += 1
                        group_means.append(scores.mean().item())
                        print(f"    {name}: Full-attn Q heads ({n_heads})", flush=True)

                    # --- Linear-attention V heads ---
                    elif "v_proj" in name and ltype == "linear_attention":
                        n_v = config.get("linear_num_value_heads", 48)
                        if n_v > 0:
                            head_dim = tensor.shape[0] // n_v
                            scores = equilibrium_group_scores(tensor, group_size=head_dim)
                            if linear_v_scores is None:
                                linear_v_scores = torch.zeros(n_v)
                            linear_v_scores += scores[:n_v]
                            linear_v_count += 1
                            group_means.append(scores.mean().item())
                            print(f"    {name}: Linear V heads ({n_v})", flush=True)

                    else:
                        scores = equilibrium_group_scores(tensor, group_size=1)
                        group_means.append(scores.mean().item())

                if group_means:
                    layer_scores[idx] = sum(group_means) / len(group_means)

    return {
        "mlp_channel_scores": mlp_channel_scores,
        "mlp_layer_count": mlp_layer_count,
        "full_head_scores": full_head_scores,
        "full_head_count": full_head_count,
        "linear_v_scores": linear_v_scores,
        "linear_v_count": linear_v_count,
        "layer_scores": layer_scores,
        "layer_types": layer_types,
    }


# ---------------------------------------------------------------------------
# Pruning plan
# ---------------------------------------------------------------------------

def generate_pruning_plan(
    scores: dict,
    config: dict,
    mlp_keep_ratio: float = 0.70,
    full_head_keep: int = 16,
    linear_v_keep: int = 34,
    layers_to_remove: int = 8,
) -> dict:
    """Turn importance scores into concrete indices to keep/remove."""
    plan: dict = {}
    num_layers = config.get("num_hidden_layers", 64)

    # Phase A — MLP channels
    if scores["mlp_channel_scores"] is not None:
        s = scores["mlp_channel_scores"]
        n_total = s.shape[0]
        n_keep = int(n_total * mlp_keep_ratio)
        if n_total >= 64:
            n_keep = max(64, (n_keep // 64) * 64)  # align to 64
        n_keep = min(n_keep, n_total)  # never exceed original size
        _, keep_idx = torch.topk(s, n_keep)
        plan["mlp_keep_channels"] = sorted(keep_idx.tolist())
        plan["mlp_new_intermediate_size"] = n_keep
        print(f"  MLP: {n_total} → {n_keep} channels ({n_keep / n_total:.1%})")

    # Phase B — Full-attention Q heads
    if scores["full_head_scores"] is not None:
        s = scores["full_head_scores"]
        n_keep = min(full_head_keep, s.shape[0])
        _, keep_idx = torch.topk(s, n_keep)
        plan["full_attn_keep_heads"] = sorted(keep_idx.tolist())
        plan["full_attn_new_num_heads"] = n_keep
        print(f"  Full-attn Q heads: {s.shape[0]} → {n_keep}")

    # Phase C — Linear V heads
    if scores["linear_v_scores"] is not None:
        s = scores["linear_v_scores"]
        n_keep = min(linear_v_keep, s.shape[0])
        _, keep_idx = torch.topk(s, n_keep)
        plan["linear_v_keep_heads"] = sorted(keep_idx.tolist())
        plan["linear_v_new_num_heads"] = n_keep
        print(f"  Linear V heads: {s.shape[0]} → {n_keep}")

    # Phase D — Layer removal (linear-attention only, protect edges)
    layer_scores = scores["layer_scores"]
    layer_types = scores["layer_types"]

    candidates = [
        (idx, sc)
        for idx, sc in layer_scores.items()
        if layer_types.get(idx) == "linear_attention"
        and PROTECTED_FIRST <= idx < (num_layers - PROTECTED_LAST)
    ]
    candidates.sort(key=lambda x: x[1])  # ascending = least important first
    n_remove = min(layers_to_remove, len(candidates))
    remove_set = sorted(c[0] for c in candidates[:n_remove])

    plan["remove_layers"] = remove_set
    plan["num_layers_after"] = num_layers - n_remove
    print(f"  Layers: {num_layers} → {num_layers - n_remove} (removing {remove_set})")

    # Layer renumbering map
    keep_layers = sorted(set(range(num_layers)) - set(remove_set))
    plan["layer_map"] = {old: new for new, old in enumerate(keep_layers)}

    return plan


# ---------------------------------------------------------------------------
# Tensor-level pruning
# ---------------------------------------------------------------------------

def prune_tensor(
    name: str,
    tensor: torch.Tensor,
    plan: dict,
    layer_idx: int,
    layer_type: str,
    config: dict,
) -> Optional[tuple[str, torch.Tensor]]:
    """Apply structural pruning to one tensor.

    Returns ``(new_name, pruned_tensor)`` or ``None`` if the layer is removed.
    """
    if layer_idx in plan.get("remove_layers", []):
        return None

    layer_map = plan.get("layer_map", {})
    new_idx = layer_map.get(layer_idx, layer_idx)
    new_name = rename_layer(name, layer_idx, new_idx)

    # Passthrough for norms / non-prunable patterns
    if any(p in name for p in SKIP_PATTERNS):
        return (new_name, tensor)

    is_bias = tensor.dim() == 1 and name.endswith(".bias")

    # Phase A — MLP channels
    keep_ch = plan.get("mlp_keep_channels")
    if keep_ch is not None:
        if "gate_proj" in name or "up_proj" in name:
            if is_bias:
                return (new_name, tensor[keep_ch])
            return (new_name, tensor[keep_ch])
        if "down_proj" in name:
            if is_bias:
                return (new_name, tensor)  # down_proj bias = hidden_size, unchanged
            return (new_name, tensor[:, keep_ch])

    # Phase B — Full-attention Q heads
    keep_heads = plan.get("full_attn_keep_heads")
    if keep_heads is not None and layer_type == "full_attention":
        n_heads = config.get("num_attention_heads", 24)

        if "q_proj" in name:
            head_dim = tensor.shape[0] // n_heads
            rows = [i for h in keep_heads for i in range(h * head_dim, (h + 1) * head_dim)]
            return (new_name, tensor[rows])

        if "o_proj" in name:
            if is_bias:
                return (new_name, tensor)  # o_proj bias = hidden_size, unchanged
            head_dim = tensor.shape[1] // n_heads
            cols = [i for h in keep_heads for i in range(h * head_dim, (h + 1) * head_dim)]
            return (new_name, tensor[:, cols])

        # k_proj / v_proj: KV heads maintained — no change

    # Phase C — Linear V heads
    keep_v = plan.get("linear_v_keep_heads")
    if keep_v is not None and layer_type == "linear_attention":
        n_v = config.get("linear_num_value_heads", 48)

        if "v_proj" in name:
            if is_bias:
                head_dim = tensor.shape[0] // n_v
                rows = [i for h in keep_v for i in range(h * head_dim, (h + 1) * head_dim)]
                return (new_name, tensor[rows])
            head_dim = tensor.shape[0] // n_v
            rows = [i for h in keep_v for i in range(h * head_dim, (h + 1) * head_dim)]
            return (new_name, tensor[rows])

        if "o_proj" in name:
            if is_bias:
                return (new_name, tensor)  # o_proj bias = hidden_size, unchanged
            head_dim = tensor.shape[1] // n_v
            cols = [i for h in keep_v for i in range(h * head_dim, (h + 1) * head_dim)]
            return (new_name, tensor[:, cols])

    return (new_name, tensor)


# ---------------------------------------------------------------------------
# Pass 2 — structural deletion
# ---------------------------------------------------------------------------

def prune_pass(
    model_dir: Path,
    output_dir: Path,
    plan: dict,
    config: dict,
    layer_types: dict[int, str],
) -> dict:
    """Read each shard, prune tensors, write to *output_dir*."""
    shards = sorted(model_dir.glob("*.safetensors"))
    stats = {"total_tensors": 0, "pruned_tensors": 0, "removed_tensors": 0}

    output_dir.mkdir(parents=True, exist_ok=True)

    for shard_i, shard in enumerate(shards):
        print(f"\n  [{shard_i + 1}/{len(shards)}] Pruning {shard.name}", flush=True)
        out_tensors: dict[str, torch.Tensor] = {}

        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                stats["total_tensors"] += 1
                idx = parse_layer_idx(name)

                if idx is None:
                    out_tensors[name] = tensor
                    continue

                ltype = layer_types.get(idx, "unknown")
                result = prune_tensor(name, tensor, plan, idx, ltype, config)

                if result is None:
                    stats["removed_tensors"] += 1
                    continue

                new_name, pruned = result
                if pruned.shape != tensor.shape:
                    stats["pruned_tensors"] += 1
                    print(
                        f"    {name}: {list(tensor.shape)} → {list(pruned.shape)}",
                        flush=True,
                    )
                out_tensors[new_name] = pruned

        if out_tensors:
            out_path = output_dir / shard.name
            save_file(out_tensors, str(out_path))
            print(f"    Wrote {out_path.name} ({len(out_tensors)} tensors)", flush=True)

    return stats


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def update_config(config: dict, plan: dict, output_dir: Path) -> dict:
    """Write an updated ``config.json`` reflecting the pruned dimensions."""
    new = config.copy()

    if "mlp_new_intermediate_size" in plan:
        new["intermediate_size"] = plan["mlp_new_intermediate_size"]

    if "full_attn_new_num_heads" in plan:
        # Preserve original head_dim so k/v proj dimensions stay consistent
        orig_heads = config.get("num_attention_heads", 1)
        head_dim = config.get("head_dim", config.get("hidden_size", 0) // orig_heads)
        new["num_attention_heads"] = plan["full_attn_new_num_heads"]
        new["head_dim"] = head_dim

    if "linear_v_new_num_heads" in plan:
        new["linear_num_value_heads"] = plan["linear_v_new_num_heads"]

    if "num_layers_after" in plan:
        new["num_hidden_layers"] = plan["num_layers_after"]

        # Rebuild layer_type list if present
        if "layer_type" in config:
            remove_set = set(plan.get("remove_layers", []))
            new["layer_type"] = [
                t for i, t in enumerate(config["layer_type"]) if i not in remove_set
            ]

    config_path = output_dir / "config.json"
    with open(config_path, "w") as fp:
        json.dump(new, fp, indent=2, ensure_ascii=False)
    print(f"  Config written: {config_path}")

    return new


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Structural pruning for Qwen3.5-27B → Q4_K_M pipeline",
    )
    parser.add_argument("--model-dir", required=True, help="Source safetensors directory")
    parser.add_argument("--output-dir", help="Output directory (default: {model-dir}-pruned)")
    parser.add_argument("--dry-run", action="store_true", help="Score and plan only")
    parser.add_argument("--mlp-keep-ratio", type=float, default=0.70)
    parser.add_argument("--full-head-keep", type=int, default=16)
    parser.add_argument("--linear-v-keep", type=int, default=34)
    parser.add_argument("--layer-remove", type=int, default=8)
    parser.add_argument("--eq-steps", type=int, default=20)
    parser.add_argument("--eq-gamma", type=float, default=0.05)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_dir.parent / f"{model_dir.name}-pruned"
    )

    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {model_dir}", file=sys.stderr)
        return 1

    with open(config_path) as fp:
        config = json.load(fp)

    banner = f"""{'=' * 60}
Structural Pruning (Equilibrium, arXiv:2512.22106)
{'=' * 60}
Model:    {model_dir}
Output:   {output_dir}
Layers:   {config.get('num_hidden_layers', '?')}
Hidden:   {config.get('hidden_size', '?')}
MLP:      {config.get('intermediate_size', '?')} → keep {args.mlp_keep_ratio:.0%}
Device:   {DEVICE}
Dry run:  {args.dry_run}
{'=' * 60}"""
    print(banner, flush=True)

    t0 = time.time()

    # Pass 1
    print(f"\n{'=' * 40}\nPass 1: Importance Scoring\n{'=' * 40}", flush=True)
    scores = score_pass(model_dir, config)
    print(f"\nPass 1 done ({time.time() - t0:.0f}s)")

    # Plan
    print(f"\n{'=' * 40}\nPruning Plan\n{'=' * 40}", flush=True)
    plan = generate_pruning_plan(
        scores,
        config,
        mlp_keep_ratio=args.mlp_keep_ratio,
        full_head_keep=args.full_head_keep,
        linear_v_keep=args.linear_v_keep,
        layers_to_remove=args.layer_remove,
    )

    # Save plan
    plan_dir = output_dir if not args.dry_run else model_dir
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_path = plan_dir / "pruning_plan.json"
    serialisable = {}
    for k, v in plan.items():
        if isinstance(v, dict):
            serialisable[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            serialisable[k] = v
    with open(plan_path, "w") as fp:
        json.dump(serialisable, fp, indent=2)
    print(f"\nPlan saved: {plan_path}")

    if args.dry_run:
        print("\nDry run complete. No model files written.")
        return 0

    # Pass 2
    t2 = time.time()
    print(f"\n{'=' * 40}\nPass 2: Structural Pruning\n{'=' * 40}", flush=True)
    stats = prune_pass(model_dir, output_dir, plan, config, scores["layer_types"])
    print(f"\nPass 2 done ({time.time() - t2:.0f}s)")

    # Config
    print(f"\n{'=' * 40}\nConfig Update\n{'=' * 40}", flush=True)
    update_config(config, plan, output_dir)

    # Copy auxiliary files
    for src in model_dir.iterdir():
        if src.suffix not in (".safetensors",) and src.name != "config.json" and src.is_file():
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    total = time.time() - t0
    print(f"\n{'=' * 60}\nDone ({total:.0f}s)\n{'=' * 60}")
    print(f"Tensors pruned: {stats['pruned_tensors']}/{stats['total_tensors']}")
    print(f"Tensors removed: {stats['removed_tensors']}")
    print(f"Output: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
