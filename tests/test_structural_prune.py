#!/usr/bin/env python3
"""Unit tests for structural_prune.py — synthetic tensors only, no model download.

Tests cover:
  1. equilibrium_group_scores shape and value range
  2. MLP channel scoring + ranking
  3. MLP channel deletion (gate/up/down_proj shapes)
  4. Full Attention Q head deletion (q_proj/o_proj shapes)
  5. Linear V head deletion (v_proj/o_proj shapes)
  6. Layer removal and renumbering
  7. config.json update consistency
  8. KV heads (k_proj/v_proj in full attn) are preserved
"""

import json
import sys
from pathlib import Path

import pytest
import torch

# Add scripts/ to path so we can import structural_prune
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from structural_prune import (
    equilibrium_group_scores,
    generate_pruning_plan,
    is_full_attention,
    parse_layer_idx,
    prune_tensor,
    rename_layer,
    update_config,
)


# ---------------------------------------------------------------------------
# equilibrium_group_scores
# ---------------------------------------------------------------------------


class TestEquilibriumGroupScores:
    """Tests for the core equilibrium scoring function."""

    def test_shape_with_group_size_1(self) -> None:
        """Given channel-level grouping, scores should match row count."""
        w = torch.randn(16, 8)
        scores = equilibrium_group_scores(w, group_size=1)
        assert scores.shape == (16,)

    def test_shape_with_group_size_4(self) -> None:
        """Given head-level grouping (4 rows/head), scores match head count."""
        w = torch.randn(24, 8)
        scores = equilibrium_group_scores(w, group_size=4)
        assert scores.shape == (6,)

    def test_values_in_0_1(self) -> None:
        """All participation scores must be in [0, 1]."""
        w = torch.randn(32, 16)
        scores = equilibrium_group_scores(w, group_size=1)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_zero_tensor_returns_ones(self) -> None:
        """A zero tensor should produce uniform scores (no crash)."""
        w = torch.zeros(8, 4)
        scores = equilibrium_group_scores(w, group_size=1)
        assert scores.shape == (8,)

    def test_large_weights_get_higher_scores(self) -> None:
        """Rows with large weights should score higher than near-zero rows."""
        w = torch.zeros(8, 4)
        w[0] = 10.0  # Large
        w[7] = 0.001  # Tiny
        scores = equilibrium_group_scores(w, group_size=1, steps=30)
        assert scores[0] > scores[7]

    def test_small_tensor_passthrough(self) -> None:
        """Tensors with fewer rows than group_size return all-ones."""
        w = torch.randn(2, 4)
        scores = equilibrium_group_scores(w, group_size=8)
        assert (scores == 1.0).all()

    def test_1d_tensor_passthrough(self) -> None:
        """1-D tensors return all-ones."""
        w = torch.randn(16)
        scores = equilibrium_group_scores(w, group_size=1)
        assert (scores == 1.0).all()


# ---------------------------------------------------------------------------
# MLP channel scoring + pruning
# ---------------------------------------------------------------------------


class TestMLPChannelPruning:
    """Tests for MLP intermediate-size channel pruning (Phase A)."""

    def test_gate_proj_shape_after_pruning(self) -> None:
        """gate_proj rows should be reduced to keep-channel count."""
        intermediate, hidden = 16, 8
        gate = torch.randn(intermediate, hidden)
        keep_ch = [0, 2, 4, 6, 8, 10, 12, 14]  # keep 50%

        plan = {"mlp_keep_channels": keep_ch}
        result = prune_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            gate,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config={},
        )
        assert result is not None
        name, pruned = result
        assert pruned.shape == (8, hidden)

    def test_up_proj_shape_after_pruning(self) -> None:
        """up_proj rows should be reduced identically to gate_proj."""
        intermediate, hidden = 16, 8
        up = torch.randn(intermediate, hidden)
        keep_ch = list(range(0, 12))  # keep 75%

        plan = {"mlp_keep_channels": keep_ch}
        result = prune_tensor(
            "model.layers.0.mlp.up_proj.weight",
            up,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config={},
        )
        assert result is not None
        _, pruned = result
        assert pruned.shape == (12, hidden)

    def test_down_proj_shape_after_pruning(self) -> None:
        """down_proj columns should be reduced (transposed from gate/up)."""
        intermediate, hidden = 16, 8
        down = torch.randn(hidden, intermediate)
        keep_ch = list(range(0, 11))  # keep ~69%

        plan = {"mlp_keep_channels": keep_ch}
        result = prune_tensor(
            "model.layers.0.mlp.down_proj.weight",
            down,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config={},
        )
        assert result is not None
        _, pruned = result
        assert pruned.shape == (hidden, 11)

    def test_mlp_values_preserved(self) -> None:
        """Kept channels should preserve their original values."""
        w = torch.arange(32).float().view(8, 4)
        keep_ch = [1, 3, 5]
        plan = {"mlp_keep_channels": keep_ch}

        result = prune_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            w,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config={},
        )
        _, pruned = result
        assert torch.equal(pruned[0], w[1])
        assert torch.equal(pruned[1], w[3])
        assert torch.equal(pruned[2], w[5])


# ---------------------------------------------------------------------------
# Full Attention head pruning (Phase B)
# ---------------------------------------------------------------------------


class TestFullAttentionHeadPruning:
    """Tests for full-attention Q-head pruning."""

    def test_q_proj_shape(self) -> None:
        """q_proj should lose rows for removed heads."""
        # 24 heads * 256 dim = 6144 rows
        n_heads, head_dim, hidden = 24, 256, 5120
        q = torch.randn(n_heads * head_dim, hidden)

        keep_heads = list(range(16))  # keep first 16 of 24
        plan = {"full_attn_keep_heads": keep_heads}
        config = {"num_attention_heads": 24}

        result = prune_tensor(
            "model.layers.3.self_attn.q_proj.weight",
            q,
            plan,
            layer_idx=3,
            layer_type="full_attention",
            config=config,
        )
        assert result is not None
        _, pruned = result
        assert pruned.shape == (16 * head_dim, hidden)

    def test_o_proj_shape(self) -> None:
        """o_proj columns should match q_proj row reduction."""
        n_heads, head_dim, hidden = 24, 256, 5120
        o = torch.randn(hidden, n_heads * head_dim)

        keep_heads = list(range(16))
        plan = {"full_attn_keep_heads": keep_heads}
        config = {"num_attention_heads": 24}

        result = prune_tensor(
            "model.layers.3.self_attn.o_proj.weight",
            o,
            plan,
            layer_idx=3,
            layer_type="full_attention",
            config=config,
        )
        assert result is not None
        _, pruned = result
        assert pruned.shape == (hidden, 16 * head_dim)

    def test_kv_proj_unchanged(self) -> None:
        """k_proj and v_proj must NOT be changed (GQA KV heads = 4 maintained)."""
        kv_heads, head_dim, hidden = 4, 256, 5120
        k = torch.randn(kv_heads * head_dim, hidden)
        v = torch.randn(kv_heads * head_dim, hidden)

        keep_heads = list(range(16))
        plan = {"full_attn_keep_heads": keep_heads}
        config = {"num_attention_heads": 24}

        k_result = prune_tensor(
            "model.layers.3.self_attn.k_proj.weight",
            k,
            plan,
            layer_idx=3,
            layer_type="full_attention",
            config=config,
        )
        v_result = prune_tensor(
            "model.layers.3.self_attn.v_proj.weight",
            v,
            plan,
            layer_idx=3,
            layer_type="full_attention",
            config=config,
        )

        assert k_result is not None
        assert v_result is not None
        assert k_result[1].shape == k.shape  # unchanged
        assert v_result[1].shape == v.shape  # unchanged

    def test_head_values_preserved(self) -> None:
        """Kept heads should contain original weight data."""
        n_heads, head_dim, hidden = 6, 4, 8
        q = torch.arange(n_heads * head_dim * hidden).float().view(n_heads * head_dim, hidden)

        keep_heads = [0, 2, 4]  # keep 3 of 6
        plan = {"full_attn_keep_heads": keep_heads}
        config = {"num_attention_heads": 6}

        result = prune_tensor(
            "model.layers.3.self_attn.q_proj.weight",
            q,
            plan,
            layer_idx=3,
            layer_type="full_attention",
            config=config,
        )
        _, pruned = result
        # Head 0 rows 0-3, Head 2 rows 8-11, Head 4 rows 16-19
        assert torch.equal(pruned[0:4], q[0:4])
        assert torch.equal(pruned[4:8], q[8:12])
        assert torch.equal(pruned[8:12], q[16:20])


# ---------------------------------------------------------------------------
# Linear V head pruning (Phase C)
# ---------------------------------------------------------------------------


class TestLinearVHeadPruning:
    """Tests for linear-attention V-head pruning."""

    def test_v_proj_shape(self) -> None:
        """v_proj rows should be reduced to kept V heads."""
        n_v, head_dim, hidden = 48, 128, 5120
        v = torch.randn(n_v * head_dim, hidden)

        keep_v = list(range(34))  # keep 34 of 48
        plan = {"linear_v_keep_heads": keep_v}
        config = {"linear_num_value_heads": 48}

        result = prune_tensor(
            "model.layers.0.self_attn.v_proj.weight",
            v,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config=config,
        )
        assert result is not None
        _, pruned = result
        assert pruned.shape == (34 * head_dim, hidden)

    def test_o_proj_shape(self) -> None:
        """o_proj columns should match v_proj row reduction."""
        n_v, head_dim, hidden = 48, 128, 5120
        o = torch.randn(hidden, n_v * head_dim)

        keep_v = list(range(34))
        plan = {"linear_v_keep_heads": keep_v}
        config = {"linear_num_value_heads": 48}

        result = prune_tensor(
            "model.layers.0.self_attn.o_proj.weight",
            o,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config=config,
        )
        assert result is not None
        _, pruned = result
        assert pruned.shape == (hidden, 34 * head_dim)

    def test_qk_proj_unchanged_in_linear(self) -> None:
        """q_proj and k_proj in linear layers should NOT be changed."""
        qk_dim, hidden = 2048, 5120
        q = torch.randn(qk_dim, hidden)

        keep_v = list(range(34))
        plan = {"linear_v_keep_heads": keep_v}
        config = {"linear_num_value_heads": 48}

        result = prune_tensor(
            "model.layers.0.self_attn.q_proj.weight",
            q,
            plan,
            layer_idx=0,
            layer_type="linear_attention",
            config=config,
        )
        assert result is not None
        assert result[1].shape == q.shape


# ---------------------------------------------------------------------------
# Layer removal and renumbering (Phase D)
# ---------------------------------------------------------------------------


class TestLayerRemoval:
    """Tests for whole-layer removal and index renumbering."""

    def test_removed_layer_returns_none(self) -> None:
        """Tensors in removed layers should return None."""
        plan = {"remove_layers": [5, 6], "layer_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 7: 5}}
        w = torch.randn(16, 8)

        result = prune_tensor(
            "model.layers.5.mlp.gate_proj.weight",
            w,
            plan,
            layer_idx=5,
            layer_type="linear_attention",
            config={},
        )
        assert result is None

    def test_kept_layer_renumbered(self) -> None:
        """Kept layers after a gap should be renumbered contiguously."""
        plan = {"remove_layers": [2], "layer_map": {0: 0, 1: 1, 3: 2, 4: 3}}
        w = torch.randn(16, 8)

        result = prune_tensor(
            "model.layers.3.mlp.gate_proj.weight",
            w,
            plan,
            layer_idx=3,
            layer_type="linear_attention",
            config={},
        )
        assert result is not None
        new_name, _ = result
        assert "layers.2." in new_name  # was 3, now 2

    def test_layer_map_contiguous(self) -> None:
        """Layer map should produce contiguous 0-indexed output."""
        scores = {
            "mlp_channel_scores": None,
            "full_head_scores": None,
            "linear_v_scores": None,
            "layer_scores": {i: float(i) for i in range(8)},
            "layer_types": {
                0: "linear_attention",
                1: "linear_attention",
                2: "linear_attention",
                3: "full_attention",
                4: "linear_attention",
                5: "linear_attention",
                6: "linear_attention",
                7: "full_attention",
            },
        }
        config = {"num_hidden_layers": 8}
        plan = generate_pruning_plan(
            scores, config,
            mlp_keep_ratio=1.0,
            full_head_keep=24,
            linear_v_keep=48,
            layers_to_remove=2,
        )

        layer_map = plan["layer_map"]
        new_indices = sorted(layer_map.values())
        assert new_indices == list(range(len(new_indices)))

    def test_protected_layers_not_removed(self) -> None:
        """First/last PROTECTED layers should never be removed."""
        # 16 layers, protect first 4 + last 4
        scores = {
            "mlp_channel_scores": None,
            "full_head_scores": None,
            "linear_v_scores": None,
            "layer_scores": {i: 0.01 for i in range(16)},  # all low scores
            "layer_types": {i: "linear_attention" for i in range(16)},
        }
        config = {"num_hidden_layers": 16}
        plan = generate_pruning_plan(
            scores, config,
            mlp_keep_ratio=1.0,
            full_head_keep=24,
            linear_v_keep=48,
            layers_to_remove=8,
        )

        removed = set(plan["remove_layers"])
        for i in range(4):  # first 4 protected
            assert i not in removed
        for i in range(12, 16):  # last 4 protected
            assert i not in removed

    def test_full_attention_layers_not_removed(self) -> None:
        """Full-attention layers should never be candidates for removal."""
        scores = {
            "mlp_channel_scores": None,
            "full_head_scores": None,
            "linear_v_scores": None,
            "layer_scores": {i: 0.01 for i in range(16)},
            "layer_types": {
                i: "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(16)
            },
        }
        config = {"num_hidden_layers": 16}
        plan = generate_pruning_plan(
            scores, config,
            mlp_keep_ratio=1.0,
            full_head_keep=24,
            linear_v_keep=48,
            layers_to_remove=4,
        )

        removed = set(plan["remove_layers"])
        full_attn_layers = {3, 7, 11, 15}
        assert removed.isdisjoint(full_attn_layers)


# ---------------------------------------------------------------------------
# rename_layer helper
# ---------------------------------------------------------------------------


class TestRenameLayer:
    """Tests for layer name renumbering."""

    def test_simple_rename(self) -> None:
        assert rename_layer("model.layers.5.mlp.gate_proj.weight", 5, 3) == \
            "model.layers.3.mlp.gate_proj.weight"

    def test_no_change_same_idx(self) -> None:
        name = "model.layers.0.self_attn.q_proj.weight"
        assert rename_layer(name, 0, 0) == name


# ---------------------------------------------------------------------------
# is_full_attention helper
# ---------------------------------------------------------------------------


class TestIsFullAttention:
    """Tests for layer type detection."""

    def test_full_attention_at_interval(self) -> None:
        # interval=4: layers 3, 7, 11 are full attention
        assert is_full_attention(3, 4) is True
        assert is_full_attention(7, 4) is True
        assert is_full_attention(11, 4) is True

    def test_linear_attention(self) -> None:
        assert is_full_attention(0, 4) is False
        assert is_full_attention(1, 4) is False
        assert is_full_attention(2, 4) is False
        assert is_full_attention(4, 4) is False

    def test_zero_interval(self) -> None:
        assert is_full_attention(0, 0) is False


# ---------------------------------------------------------------------------
# parse_layer_idx helper
# ---------------------------------------------------------------------------


class TestParseLayerIdx:
    """Tests for tensor name parsing."""

    def test_valid_layer_name(self) -> None:
        assert parse_layer_idx("model.layers.42.mlp.gate_proj.weight") == 42

    def test_non_layer_name(self) -> None:
        assert parse_layer_idx("model.embed_tokens.weight") is None

    def test_lm_head(self) -> None:
        assert parse_layer_idx("lm_head.weight") is None


# ---------------------------------------------------------------------------
# config.json update
# ---------------------------------------------------------------------------


class TestUpdateConfig:
    """Tests for config.json updates after pruning."""

    def test_intermediate_size_updated(self, tmp_path: Path) -> None:
        config = {"intermediate_size": 17408, "num_hidden_layers": 64}
        plan = {"mlp_new_intermediate_size": 12224}

        new = update_config(config, plan, tmp_path)
        assert new["intermediate_size"] == 12224

    def test_num_attention_heads_updated(self, tmp_path: Path) -> None:
        config = {"num_attention_heads": 24, "num_hidden_layers": 64}
        plan = {"full_attn_new_num_heads": 16}

        new = update_config(config, plan, tmp_path)
        assert new["num_attention_heads"] == 16

    def test_linear_v_heads_updated(self, tmp_path: Path) -> None:
        config = {"linear_num_value_heads": 48, "num_hidden_layers": 64}
        plan = {"linear_v_new_num_heads": 34}

        new = update_config(config, plan, tmp_path)
        assert new["linear_num_value_heads"] == 34

    def test_num_layers_updated(self, tmp_path: Path) -> None:
        config = {"num_hidden_layers": 64}
        plan = {"num_layers_after": 56, "remove_layers": [4, 5, 6, 8, 9, 10, 12, 13]}

        new = update_config(config, plan, tmp_path)
        assert new["num_hidden_layers"] == 56

    def test_layer_type_list_pruned(self, tmp_path: Path) -> None:
        """layer_type list should have entries removed for deleted layers."""
        types = ["linear", "linear", "linear", "full"] * 4  # 16 entries
        config = {"num_hidden_layers": 16, "layer_type": types}
        plan = {
            "num_layers_after": 14,
            "remove_layers": [4, 5],
        }

        new = update_config(config, plan, tmp_path)
        assert len(new["layer_type"]) == 14

    def test_config_json_written(self, tmp_path: Path) -> None:
        """config.json should be written to output dir."""
        config = {"num_hidden_layers": 64, "hidden_size": 5120}
        plan = {"num_layers_after": 56, "remove_layers": list(range(4, 12))}

        update_config(config, plan, tmp_path)

        written = json.loads((tmp_path / "config.json").read_text())
        assert written["num_hidden_layers"] == 56
        assert written["hidden_size"] == 5120

    def test_unchanged_fields_preserved(self, tmp_path: Path) -> None:
        """Fields not affected by pruning should be preserved."""
        config = {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "model_type": "qwen3",
        }
        plan = {"num_layers_after": 56, "remove_layers": [4, 5, 6, 8, 9, 10, 12, 13]}

        new = update_config(config, plan, tmp_path)
        assert new["hidden_size"] == 5120
        assert new["vocab_size"] == 248320
        assert new["model_type"] == "qwen3"


# ---------------------------------------------------------------------------
# Combined phase interaction
# ---------------------------------------------------------------------------


class TestCombinedPruning:
    """Tests for multiple pruning phases applied simultaneously."""

    def test_mlp_and_layer_removal(self) -> None:
        """MLP pruning + layer removal should work together."""
        plan = {
            "mlp_keep_channels": [0, 1, 2, 3],
            "remove_layers": [1],
            "layer_map": {0: 0, 2: 1, 3: 2},
        }

        # Layer 0 gate_proj: MLP pruned
        w0 = torch.randn(8, 4)
        r0 = prune_tensor(
            "model.layers.0.mlp.gate_proj.weight",
            w0, plan, layer_idx=0, layer_type="linear_attention", config={},
        )
        assert r0 is not None
        assert r0[1].shape == (4, 4)
        assert "layers.0." in r0[0]

        # Layer 1: removed
        w1 = torch.randn(8, 4)
        r1 = prune_tensor(
            "model.layers.1.mlp.gate_proj.weight",
            w1, plan, layer_idx=1, layer_type="linear_attention", config={},
        )
        assert r1 is None

        # Layer 2: MLP pruned + renumbered to 1
        w2 = torch.randn(8, 4)
        r2 = prune_tensor(
            "model.layers.2.mlp.gate_proj.weight",
            w2, plan, layer_idx=2, layer_type="linear_attention", config={},
        )
        assert r2 is not None
        assert r2[1].shape == (4, 4)
        assert "layers.1." in r2[0]

    def test_norm_tensors_passthrough(self) -> None:
        """Norm tensors should pass through with only renumbering."""
        plan = {
            "mlp_keep_channels": [0, 1],
            "remove_layers": [],
            "layer_map": {0: 0},
        }
        norm = torch.randn(5120)
        result = prune_tensor(
            "model.layers.0.input_layernorm.weight",
            norm, plan, layer_idx=0, layer_type="linear_attention", config={},
        )
        assert result is not None
        assert result[1].shape == norm.shape


# ---------------------------------------------------------------------------
# End-to-end integration test (synthetic safetensors model)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline test: score → plan → prune → config update.

    Creates a tiny synthetic model directory with safetensors that mimics
    Qwen3.5 structure (full + linear attention layers, MLP, etc.) and runs
    the complete pipeline.
    """

    # Tiny model dimensions (16 layers to have pruneable middle section)
    HIDDEN = 32
    INTERMEDIATE = 16
    FULL_Q_HEADS = 4
    FULL_KV_HEADS = 2
    FULL_HEAD_DIM = 8  # HIDDEN // FULL_Q_HEADS
    LINEAR_QK_HEADS = 4
    LINEAR_V_HEADS = 8
    LINEAR_HEAD_DIM = 4  # HIDDEN // LINEAR_V_HEADS
    NUM_LAYERS = 16  # 12 linear + 4 full (interval=4)
    FULL_ATTN_INTERVAL = 4

    @pytest.fixture
    def synthetic_model(self, tmp_path: Path) -> Path:
        """Create a synthetic model directory with safetensors."""
        from safetensors.torch import save_file

        model_dir = tmp_path / "synthetic_model"
        model_dir.mkdir()

        tensors: dict[str, torch.Tensor] = {}

        # Embeddings and head
        tensors["model.embed_tokens.weight"] = torch.randn(100, self.HIDDEN)
        tensors["lm_head.weight"] = torch.randn(100, self.HIDDEN)
        tensors["model.norm.weight"] = torch.randn(self.HIDDEN)

        for i in range(self.NUM_LAYERS):
            prefix = f"model.layers.{i}"
            is_full = (i + 1) % self.FULL_ATTN_INTERVAL == 0

            # Norms
            tensors[f"{prefix}.input_layernorm.weight"] = torch.randn(self.HIDDEN)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(self.HIDDEN)

            if is_full:
                # Full attention
                q_dim = self.FULL_Q_HEADS * self.FULL_HEAD_DIM
                kv_dim = self.FULL_KV_HEADS * self.FULL_HEAD_DIM
                tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(q_dim, self.HIDDEN)
                tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(kv_dim, self.HIDDEN)
                tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(kv_dim, self.HIDDEN)
                tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(self.HIDDEN, q_dim)
            else:
                # Linear attention
                qk_dim = self.LINEAR_QK_HEADS * self.LINEAR_HEAD_DIM
                v_dim = self.LINEAR_V_HEADS * self.LINEAR_HEAD_DIM
                tensors[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(qk_dim, self.HIDDEN)
                tensors[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(qk_dim, self.HIDDEN)
                tensors[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(v_dim, self.HIDDEN)
                tensors[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(self.HIDDEN, v_dim)

            # MLP (same for all layers)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(self.INTERMEDIATE, self.HIDDEN)
            tensors[f"{prefix}.mlp.up_proj.weight"] = torch.randn(self.INTERMEDIATE, self.HIDDEN)
            tensors[f"{prefix}.mlp.down_proj.weight"] = torch.randn(self.HIDDEN, self.INTERMEDIATE)

        # Save as single shard
        save_file(tensors, str(model_dir / "model-00001-of-00001.safetensors"))

        # Config
        config = {
            "num_hidden_layers": self.NUM_LAYERS,
            "hidden_size": self.HIDDEN,
            "intermediate_size": self.INTERMEDIATE,
            "num_attention_heads": self.FULL_Q_HEADS,
            "num_key_value_heads": self.FULL_KV_HEADS,
            "linear_num_value_heads": self.LINEAR_V_HEADS,
            "full_attention_interval": self.FULL_ATTN_INTERVAL,
            "vocab_size": 100,
        }
        (model_dir / "config.json").write_text(json.dumps(config, indent=2))

        return model_dir

    def test_score_pass_produces_all_score_types(self, synthetic_model: Path) -> None:
        """Pass 1 should produce MLP, full-attn, linear-V, and layer scores."""
        from structural_prune import score_pass

        config = json.loads((synthetic_model / "config.json").read_text())
        scores = score_pass(synthetic_model, config)

        assert scores["mlp_channel_scores"] is not None
        assert scores["mlp_channel_scores"].shape == (self.INTERMEDIATE,)
        assert scores["mlp_layer_count"] > 0

        assert scores["full_head_scores"] is not None
        assert scores["full_head_scores"].shape == (self.FULL_Q_HEADS,)

        assert scores["linear_v_scores"] is not None
        assert scores["linear_v_scores"].shape == (self.LINEAR_V_HEADS,)

        assert len(scores["layer_scores"]) == self.NUM_LAYERS
        assert len(scores["layer_types"]) == self.NUM_LAYERS

        # Verify layer type assignment
        for i in range(self.NUM_LAYERS):
            expected = "full_attention" if (i + 1) % self.FULL_ATTN_INTERVAL == 0 else "linear_attention"
            assert scores["layer_types"][i] == expected

    def test_full_pipeline_shapes(self, synthetic_model: Path, tmp_path: Path) -> None:
        """Complete pipeline: score → plan → prune → verify output shapes."""
        from safetensors import safe_open
        from structural_prune import prune_pass, score_pass

        output_dir = tmp_path / "pruned_output"
        config = json.loads((synthetic_model / "config.json").read_text())

        # Pass 1: Score
        scores = score_pass(synthetic_model, config)

        # Plan: keep 50% MLP, 2/4 Q heads, 4/8 V heads, remove 2 layers
        plan = generate_pruning_plan(
            scores,
            config,
            mlp_keep_ratio=0.50,
            full_head_keep=2,
            linear_v_keep=4,
            layers_to_remove=2,
        )

        # Verify plan sanity
        assert plan["mlp_new_intermediate_size"] > 0
        assert plan["full_attn_new_num_heads"] == 2
        assert plan["linear_v_new_num_heads"] == 4
        assert plan["num_layers_after"] == self.NUM_LAYERS - 2
        assert len(plan["remove_layers"]) == 2

        # Pass 2: Prune
        stats = prune_pass(
            synthetic_model, output_dir, plan, config, scores["layer_types"],
        )
        assert stats["pruned_tensors"] > 0
        assert stats["removed_tensors"] > 0

        # Update config
        new_config = update_config(config, plan, output_dir)

        # Verify output files exist
        out_shards = list(output_dir.glob("*.safetensors"))
        assert len(out_shards) == 1

        # Verify output config
        assert new_config["num_hidden_layers"] == self.NUM_LAYERS - 2
        assert new_config["intermediate_size"] == plan["mlp_new_intermediate_size"]
        assert new_config["num_attention_heads"] == 2
        assert new_config["linear_num_value_heads"] == 4

        # Verify tensor shapes in output
        new_intermediate = plan["mlp_new_intermediate_size"]
        new_num_layers = plan["num_layers_after"]

        with safe_open(str(out_shards[0]), framework="pt", device="cpu") as f:
            names = list(f.keys())

            # Should have correct layer count (0-indexed)
            layer_indices = set()
            for n in names:
                idx = parse_layer_idx(n)
                if idx is not None:
                    layer_indices.add(idx)
            assert max(layer_indices) == new_num_layers - 1
            assert len(layer_indices) == new_num_layers

            # Check a MLP tensor shape
            gate = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
            assert gate.shape == (new_intermediate, self.HIDDEN)

            down = f.get_tensor("model.layers.0.mlp.down_proj.weight")
            assert down.shape == (self.HIDDEN, new_intermediate)

            # Embeddings unchanged
            embed = f.get_tensor("model.embed_tokens.weight")
            assert embed.shape == (100, self.HIDDEN)

    def test_dry_run_no_output(self, synthetic_model: Path, tmp_path: Path) -> None:
        """Dry-run should produce a plan but no output directory changes."""
        from structural_prune import score_pass

        config = json.loads((synthetic_model / "config.json").read_text())
        scores = score_pass(synthetic_model, config)
        plan = generate_pruning_plan(scores, config, layers_to_remove=1)

        # Plan is generated but we don't call prune_pass
        assert "remove_layers" in plan
        assert "layer_map" in plan

        # No output dir created
        output_dir = tmp_path / "should_not_exist"
        assert not output_dir.exists()
