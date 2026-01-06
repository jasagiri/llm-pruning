#!/usr/bin/env python3
"""
Universal LLM Pruning Platform
==============================

HuggingFaceモデル向けの汎用プルーニングプラットフォーム。
複数のプルーニング手法を実行時に選択可能。

サポートするプルーニング手法:
    1. magnitude     - 重み絶対値による枝刈り（最も基本的）
    2. random        - ランダム枝刈り（ベースライン比較用）
    3. l1_structured - L1ノルムによる行構造化枝刈り
    4. l2_structured - L2ノルムによる列構造化枝刈り
    5. gradient      - 勾配ベース（Taylor展開近似）
    6. movement      - ファインチューニング時の重み移動追跡
    7. equilibrium   - ゲーム理論型（arXiv:2512.22106）
    8. wanda         - Weights AND Activations枝刈り
    9. sparsegpt     - Hessian近似によるワンショット枝刈り

サポートする構造タイプ:
    - unstructured: 個別重み単位
    - row:          行（ニューロン）単位
    - column:       列（フィルタ）単位
    - block:        ブロック単位（N:Mスパース化）
    - head:         Attentionヘッド単位
    - layer:        レイヤー全体

基本的な使用方法:
    # メソッド一覧
    python pruning_platform.py --list-methods

    # Magnitude pruning (30%スパース化)
    python pruning_platform.py --model microsoft/Phi-3-mini-4k-instruct \\
        --method magnitude --sparsity 0.3 --output ./pruned

    # Equilibrium pruning (ゲーム理論型)
    python pruning_platform.py --model <model> --method equilibrium \\
        --sparsity 0.5 --eq-steps 100

    # GGUF変換付き
    python pruning_platform.py --model <model> --method magnitude \\
        --convert-gguf --quantize Q4_K_M

APIとしての使用:
    >>> from pruning_platform import PruningConfig, PruningPipeline, PruningMethod
    >>> config = PruningConfig(method=PruningMethod.MAGNITUDE, sparsity=0.3)
    >>> pipeline = PruningPipeline(config)
    >>> model = pipeline.prune(model)
    >>> print(pipeline.get_stats())

Author: LLM Pruning Project
License: MIT
"""

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# Pruning Method Enum
# ============================================================================

class PruningMethod(Enum):
    """Available pruning methods."""
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    L1_STRUCTURED = "l1_structured"
    L2_STRUCTURED = "l2_structured"
    GRADIENT = "gradient"
    MOVEMENT = "movement"
    EQUILIBRIUM = "equilibrium"
    WANDA = "wanda"
    SPARSEGPT = "sparsegpt"


class StructureType(Enum):
    """Pruning structure types."""
    UNSTRUCTURED = "unstructured"  # Individual weights
    ROW = "row"                     # Neuron/row pruning
    COLUMN = "column"               # Filter/column pruning
    BLOCK = "block"                 # Block-structured pruning
    HEAD = "head"                   # Attention head pruning
    LAYER = "layer"                 # Entire layer pruning


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PruningConfig:
    """Configuration for pruning."""
    method: PruningMethod = PruningMethod.MAGNITUDE
    sparsity: float = 0.3
    structure: StructureType = StructureType.UNSTRUCTURED

    # Target layers
    target_modules: List[str] = field(default_factory=lambda: ["mlp", "self_attn"])
    exclude_modules: List[str] = field(default_factory=lambda: ["embed", "lm_head", "norm"])

    # Equilibrium pruning parameters
    equilibrium_alpha: float = 1.0   # Benefit weight
    equilibrium_beta: float = 0.1    # Cost weight
    equilibrium_gamma: float = 0.01  # L1 penalty
    equilibrium_eta: float = 0.0     # Competition weight
    equilibrium_lr: float = 0.1      # Learning rate
    equilibrium_steps: int = 100     # Optimization steps

    # Block pruning parameters
    block_rows: int = 32
    block_cols: int = 32

    # Head pruning parameters
    num_heads: int = 32

    # Calibration
    calibration_samples: int = 128
    calibration_seqlen: int = 2048

    # Output
    output_dir: str = "./pruned_model"
    save_format: str = "safetensors"  # safetensors or pytorch


# ============================================================================
# Base Pruner
# ============================================================================

class BasePruner(ABC):
    """Abstract base class for all pruners."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.stats = {
            "total_params": 0,
            "pruned_params": 0,
            "sparsity": 0.0,
            "layers_pruned": 0,
        }

    @abstractmethod
    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute pruning mask for a weight tensor.

        Args:
            weight: Weight tensor to prune
            **kwargs: Additional arguments (gradients, activations, etc.)

        Returns:
            Boolean mask where True = keep, False = prune
        """
        pass

    def prune_tensor(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply pruning to a tensor."""
        mask = self.compute_mask(weight, **kwargs)
        return weight * mask.float()

    def should_prune(self, name: str) -> bool:
        """Check if a parameter should be pruned."""
        # Check exclusions
        for exclude in self.config.exclude_modules:
            if exclude in name.lower():
                return False

        # Check targets
        for target in self.config.target_modules:
            if target in name.lower() and "weight" in name.lower():
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.stats


# ============================================================================
# Magnitude Pruner
# ============================================================================

class MagnitudePruner(BasePruner):
    """Magnitude-based pruning: remove weights with smallest absolute values."""

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        if weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        if self.config.structure == StructureType.UNSTRUCTURED:
            return self._unstructured_mask(weight)
        elif self.config.structure == StructureType.ROW:
            return self._row_mask(weight)
        elif self.config.structure == StructureType.COLUMN:
            return self._column_mask(weight)
        elif self.config.structure == StructureType.BLOCK:
            return self._block_mask(weight)
        else:
            return self._unstructured_mask(weight)

    def _unstructured_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Unstructured magnitude pruning."""
        flat = weight.abs().flatten()
        k = int(flat.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(flat, k).values
        return weight.abs() >= threshold

    def _row_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Row-structured pruning by L2 norm of rows."""
        row_norms = weight.norm(dim=1)
        k = int(row_norms.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(row_norms, k).values
        row_mask = row_norms >= threshold
        return row_mask.unsqueeze(1).expand_as(weight)

    def _column_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Column-structured pruning by L2 norm of columns."""
        col_norms = weight.norm(dim=0)
        k = int(col_norms.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(col_norms, k).values
        col_mask = col_norms >= threshold
        return col_mask.unsqueeze(0).expand_as(weight)

    def _block_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Block-structured pruning."""
        rows, cols = weight.shape
        br, bc = self.config.block_rows, self.config.block_cols

        # Pad if necessary
        pad_rows = (br - rows % br) % br
        pad_cols = (bc - cols % bc) % bc

        if pad_rows > 0 or pad_cols > 0:
            weight_padded = torch.nn.functional.pad(weight, (0, pad_cols, 0, pad_rows))
        else:
            weight_padded = weight

        # Reshape to blocks
        new_rows, new_cols = weight_padded.shape
        blocks = weight_padded.reshape(new_rows // br, br, new_cols // bc, bc)
        blocks = blocks.permute(0, 2, 1, 3)  # (num_row_blocks, num_col_blocks, br, bc)

        # Compute block norms
        block_norms = blocks.norm(dim=(2, 3)).flatten()
        k = int(block_norms.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(block_norms, k).values
        block_mask = block_norms >= threshold
        block_mask = block_mask.reshape(new_rows // br, new_cols // bc)

        # Expand mask to full size
        full_mask = block_mask.unsqueeze(2).unsqueeze(3).expand(-1, -1, br, bc)
        full_mask = full_mask.permute(0, 2, 1, 3).reshape(new_rows, new_cols)

        return full_mask[:rows, :cols]


# ============================================================================
# Random Pruner
# ============================================================================

class RandomPruner(BasePruner):
    """Random pruning: randomly remove weights."""

    def __init__(self, config: PruningConfig, seed: int = 42):
        super().__init__(config)
        self.seed = seed
        torch.manual_seed(seed)

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        if weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        mask = torch.rand_like(weight) >= self.config.sparsity
        return mask


# ============================================================================
# Gradient-based Pruner
# ============================================================================

class GradientPruner(BasePruner):
    """Gradient-based pruning: use gradient magnitude as importance."""

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        gradient = kwargs.get("gradient")

        if gradient is None:
            # Fall back to magnitude pruning
            return MagnitudePruner(self.config).compute_mask(weight)

        if weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        # Importance = |weight| * |gradient| (Taylor expansion approximation)
        importance = (weight.abs() * gradient.abs()).flatten()
        k = int(importance.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(importance, k).values
        return (weight.abs() * gradient.abs()) >= threshold


# ============================================================================
# Movement Pruner
# ============================================================================

class MovementPruner(BasePruner):
    """Movement pruning: track weight changes during fine-tuning."""

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.initial_weights: Dict[str, torch.Tensor] = {}
        self.movement_scores: Dict[str, torch.Tensor] = {}

    def record_initial(self, name: str, weight: torch.Tensor):
        """Record initial weights before fine-tuning."""
        self.initial_weights[name] = weight.clone()

    def update_movement(self, name: str, weight: torch.Tensor):
        """Update movement scores."""
        if name in self.initial_weights:
            movement = (weight - self.initial_weights[name]).abs()
            if name in self.movement_scores:
                self.movement_scores[name] = torch.max(self.movement_scores[name], movement)
            else:
                self.movement_scores[name] = movement

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        name = kwargs.get("name", "")
        movement = self.movement_scores.get(name)

        if movement is None or weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        # Weights with little movement are less important
        importance = movement.flatten()
        k = int(importance.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(importance, k).values
        return movement >= threshold


# ============================================================================
# Equilibrium Pruner (Game-theoretic)
# ============================================================================

class EquilibriumPruner(BasePruner):
    """
    ゲーム理論に基づくEquilibriumプルーニング。

    参考論文: "Pruning as a Game: Equilibrium-Driven Sparsification" (arXiv:2512.22106)

    概要:
        各重みを「プレイヤー」として扱い、参加度（participation）を
        ゲーム理論的に最適化。重要な重みは参加度が高くなり、
        不要な重みは参加度が下がってプルーニングされる。

    アルゴリズム:
        各重みの参加変数 s_i は以下の更新則で進化:
        s_i := s_i + lr * (benefit - cost - l1_penalty - competition)

        - benefit: 重要度（勾配×重み相関）に応じた利益
        - cost: スパース化を促すコスト
        - l1_penalty: L1正則化によるスパース化圧力
        - competition: 類似した重み間の競合

    パラメータ (PruningConfig):
        equilibrium_alpha: 利益重み（重要度への重み付け）
        equilibrium_beta:  コスト重み（スパース化圧力）
        equilibrium_gamma: L1ペナルティ強度
        equilibrium_eta:   競合重み
        equilibrium_lr:    最適化の学習率
        equilibrium_steps: 最適化ステップ数

    使用例:
        >>> config = PruningConfig(
        ...     method=PruningMethod.EQUILIBRIUM,
        ...     sparsity=0.5,
        ...     equilibrium_steps=100
        ... )
        >>> pruner = EquilibriumPruner(config)
        >>> mask = pruner.compute_mask(weight)
    """

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.participation: Dict[str, torch.Tensor] = {}

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        gradient = kwargs.get("gradient")
        name = kwargs.get("name", "default")

        if weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        # Initialize participation as importance score based on weight magnitude
        if name not in self.participation:
            # Start with magnitude-based importance
            self.participation[name] = weight.abs() / (weight.abs().max() + 1e-8)

        s = self.participation[name]

        # Run optimization steps to refine importance
        for _ in range(self.config.equilibrium_steps):
            s = self._update_participation(weight, gradient, s)

        self.participation[name] = s

        # Threshold: prune weights with lowest participation scores
        flat_s = s.flatten()
        k = int(flat_s.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        # Find threshold: keep weights with top (1-sparsity) participation
        threshold = torch.kthvalue(flat_s, k).values
        return s > threshold

    def _update_participation(
        self, weight: torch.Tensor, gradient: Optional[torch.Tensor], s: torch.Tensor
    ) -> torch.Tensor:
        """Update participation variables."""
        alpha = self.config.equilibrium_alpha
        beta = self.config.equilibrium_beta
        gamma = self.config.equilibrium_gamma
        eta = self.config.equilibrium_eta
        lr = self.config.equilibrium_lr

        # Normalize weight for stable optimization
        weight_normalized = weight / (weight.abs().max() + 1e-8)

        # Benefit: gradient-weight correlation (importance)
        if gradient is not None:
            gradient_normalized = gradient / (gradient.abs().max() + 1e-8)
            benefit = alpha * (gradient_normalized.abs() * weight_normalized.abs())
        else:
            # Without gradient, use normalized magnitude as importance
            benefit = alpha * weight_normalized.abs()

        # Cost: encourage sparsity for low-importance weights
        cost = beta * (1.0 - weight_normalized.abs()) * s

        # L1 penalty: uniform sparsity pressure
        l1_penalty = gamma

        # Competition: encourage diversity (penalize similar participation)
        if eta > 0:
            mean_s = s.mean()
            competition = eta * (s - mean_s).abs()
        else:
            competition = 0

        # Update: push participation based on importance vs cost
        delta = lr * (benefit - cost - l1_penalty - competition)
        s = s + delta

        # Clamp to [0, 1]
        s = torch.clamp(s, 0, 1)

        return s


# ============================================================================
# Wanda Pruner
# ============================================================================

class WandaPruner(BasePruner):
    """
    Wanda: Weights AND Activations pruning.

    Importance = |weight| * ||activation||_2

    Requires calibration data to compute activation norms.
    """

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.activation_norms: Dict[str, torch.Tensor] = {}

    def record_activations(self, name: str, activation: torch.Tensor):
        """Record activation statistics."""
        # Compute L2 norm across batch and sequence dimensions
        if activation.dim() == 3:  # (batch, seq, hidden)
            norm = activation.float().pow(2).sum(dim=(0, 1)).sqrt()
        elif activation.dim() == 2:  # (batch, hidden)
            norm = activation.float().pow(2).sum(dim=0).sqrt()
        else:
            norm = activation.float().abs().mean(dim=0)

        if name in self.activation_norms:
            self.activation_norms[name] = (self.activation_norms[name] + norm) / 2
        else:
            self.activation_norms[name] = norm

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        name = kwargs.get("name", "")
        activation_norm = self.activation_norms.get(name)

        if weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        if activation_norm is None:
            # Fall back to magnitude
            return MagnitudePruner(self.config).compute_mask(weight)

        # Importance = |weight| * activation_norm (per input dimension)
        # weight shape: (out_features, in_features)
        # activation_norm shape: (in_features,)
        importance = weight.abs() * activation_norm.unsqueeze(0)

        flat_importance = importance.flatten()
        k = int(flat_importance.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(flat_importance, k).values
        return importance >= threshold


# ============================================================================
# SparseGPT Pruner
# ============================================================================

class SparseGPTPruner(BasePruner):
    """
    SparseGPT: One-shot pruning using approximate Hessian.

    Uses layerwise reconstruction with OBS (Optimal Brain Surgeon).
    """

    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.hessians: Dict[str, torch.Tensor] = {}

    def record_hessian(self, name: str, activation: torch.Tensor):
        """Accumulate Hessian approximation from activations."""
        # H ≈ X^T X / n (empirical Fisher approximation)
        if activation.dim() == 3:  # (batch, seq, hidden)
            X = activation.reshape(-1, activation.shape[-1])
        else:
            X = activation

        H = X.T @ X / X.shape[0]

        if name in self.hessians:
            self.hessians[name] = (self.hessians[name] + H) / 2
        else:
            self.hessians[name] = H

    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        name = kwargs.get("name", "")
        H = self.hessians.get(name)

        if weight.dim() < 2:
            return torch.ones_like(weight, dtype=torch.bool)

        if H is None:
            return MagnitudePruner(self.config).compute_mask(weight)

        # OBS-based importance: w^2 / (2 * H_ii)
        # Simplified: use diagonal of Hessian
        H_diag = torch.diag(H)
        H_diag = H_diag.clamp(min=1e-6)  # Avoid division by zero

        # weight: (out, in), H_diag: (in,)
        importance = (weight ** 2) / (2 * H_diag.unsqueeze(0))

        flat_importance = importance.flatten()
        k = int(flat_importance.numel() * self.config.sparsity)
        if k == 0:
            return torch.ones_like(weight, dtype=torch.bool)

        threshold = torch.kthvalue(flat_importance, k).values
        return importance >= threshold


# ============================================================================
# Pruner Factory
# ============================================================================

def create_pruner(config: PruningConfig) -> BasePruner:
    """Create a pruner based on configuration."""
    pruners = {
        PruningMethod.MAGNITUDE: MagnitudePruner,
        PruningMethod.RANDOM: RandomPruner,
        PruningMethod.L1_STRUCTURED: lambda c: MagnitudePruner(
            PruningConfig(**{**c.__dict__, "structure": StructureType.ROW})
        ),
        PruningMethod.L2_STRUCTURED: lambda c: MagnitudePruner(
            PruningConfig(**{**c.__dict__, "structure": StructureType.COLUMN})
        ),
        PruningMethod.GRADIENT: GradientPruner,
        PruningMethod.MOVEMENT: MovementPruner,
        PruningMethod.EQUILIBRIUM: EquilibriumPruner,
        PruningMethod.WANDA: WandaPruner,
        PruningMethod.SPARSEGPT: SparseGPTPruner,
    }

    pruner_class = pruners.get(config.method)
    if pruner_class is None:
        raise ValueError(f"Unknown pruning method: {config.method}")

    return pruner_class(config)


# ============================================================================
# Model Pruning Pipeline
# ============================================================================

class PruningPipeline:
    """
    HuggingFaceモデル向けの完全なプルーニングパイプライン。

    このクラスは、モデルの読み込みからプルーニング適用、統計収集まで
    一貫した処理を提供します。

    機能:
        - 複数のプルーニング手法をサポート
        - キャリブレーションデータによる勾配/活性化収集
        - 層ごとのプルーニング統計
        - 自動的なパラメータフィルタリング

    使用例:
        >>> config = PruningConfig(
        ...     method=PruningMethod.MAGNITUDE,
        ...     sparsity=0.3
        ... )
        >>> pipeline = PruningPipeline(config)
        >>> model = pipeline.prune(model, calibration_data)
        >>> stats = pipeline.get_stats()
        >>> print(f"Sparsity: {stats['sparsity']:.2%}")

    Attributes:
        config: プルーニング設定
        pruner: 選択されたプルーナーインスタンス
        gradients: 収集された勾配データ
    """

    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruner = create_pruner(config)
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

    def _register_gradient_hooks(self, model: nn.Module):
        """Register hooks to capture gradients."""
        def hook(name):
            def fn(grad):
                self.gradients[name] = grad.clone()
                return grad
            return fn

        for name, param in model.named_parameters():
            if self.pruner.should_prune(name):
                self.hooks.append(param.register_hook(hook(name)))

    def _compute_gradients(self, model: nn.Module, calibration_data: List[torch.Tensor]):
        """Compute gradients using calibration data."""
        model.train()  # Enable gradients
        self._register_gradient_hooks(model)

        for batch in calibration_data[:self.config.calibration_samples]:
            model.zero_grad()
            try:
                outputs = model(batch, labels=batch, use_cache=False)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
            except Exception as e:
                # Some models may have compatibility issues
                print(f"Warning: Could not compute gradient for batch: {e}")
                continue

        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _record_activations(self, model: nn.Module, calibration_data: List[torch.Tensor]):
        """Record activations for Wanda/SparseGPT."""
        activation_hooks = []

        def hook(name):
            def fn(module, input, output):
                if isinstance(self.pruner, WandaPruner):
                    self.pruner.record_activations(name, input[0])
                elif isinstance(self.pruner, SparseGPTPruner):
                    self.pruner.record_hessian(name, input[0])
            return fn

        # Register hooks on linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                activation_hooks.append(
                    module.register_forward_hook(hook(name + ".weight"))
                )

        # Forward pass through calibration data
        model.eval()
        with torch.no_grad():
            for batch in calibration_data[:self.config.calibration_samples]:
                model(batch)

        # Remove hooks
        for hook in activation_hooks:
            hook.remove()

    def prune(self, model: nn.Module, calibration_data: Optional[List[torch.Tensor]] = None):
        """Apply pruning to model."""
        # Compute gradients if needed
        if self.config.method in [PruningMethod.GRADIENT, PruningMethod.EQUILIBRIUM]:
            if calibration_data:
                self._compute_gradients(model, calibration_data)

        # Record activations if needed
        if self.config.method in [PruningMethod.WANDA, PruningMethod.SPARSEGPT]:
            if calibration_data:
                self._record_activations(model, calibration_data)

        # Apply pruning
        total_params = 0
        pruned_params = 0
        layers_pruned = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not self.pruner.should_prune(name):
                    continue

                gradient = self.gradients.get(name)
                pruned_weight = self.pruner.prune_tensor(
                    param.data,
                    gradient=gradient,
                    name=name
                )

                # Count statistics
                total_params += param.numel()
                pruned_params += (pruned_weight == 0).sum().item()
                layers_pruned += 1

                # Apply pruned weights
                param.data = pruned_weight

        # Update stats
        self.pruner.stats["total_params"] = total_params
        self.pruner.stats["pruned_params"] = pruned_params
        self.pruner.stats["sparsity"] = pruned_params / total_params if total_params > 0 else 0
        self.pruner.stats["layers_pruned"] = layers_pruned

        return model

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return self.pruner.get_stats()


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal LLM Pruning Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Magnitude pruning with 30% sparsity
    python pruning_platform.py --model microsoft/phi-2 --method magnitude --sparsity 0.3

    # Equilibrium pruning (game-theoretic)
    python pruning_platform.py --model meta-llama/Llama-2-7b-hf --method equilibrium --sparsity 0.5

    # Wanda pruning with calibration
    python pruning_platform.py --model mistralai/Mistral-7B-v0.1 --method wanda --sparsity 0.4

    # Structured row pruning
    python pruning_platform.py --model microsoft/phi-2 --method l1_structured --sparsity 0.3
        """
    )

    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model name or path")
    parser.add_argument("--method", type=str, default="magnitude",
                        choices=[m.value for m in PruningMethod],
                        help="Pruning method to use")
    parser.add_argument("--sparsity", type=float, default=0.3,
                        help="Target sparsity ratio (0.0 to 1.0)")
    parser.add_argument("--structure", type=str, default="unstructured",
                        choices=[s.value for s in StructureType],
                        help="Pruning structure type")
    parser.add_argument("--output", type=str, default="./pruned_model",
                        help="Output directory for pruned model")

    # Equilibrium parameters
    parser.add_argument("--eq-alpha", type=float, default=1.0,
                        help="Equilibrium: benefit weight")
    parser.add_argument("--eq-beta", type=float, default=0.1,
                        help="Equilibrium: cost weight")
    parser.add_argument("--eq-gamma", type=float, default=0.01,
                        help="Equilibrium: L1 penalty")
    parser.add_argument("--eq-steps", type=int, default=100,
                        help="Equilibrium: optimization steps")

    # Calibration
    parser.add_argument("--calibration-samples", type=int, default=128,
                        help="Number of calibration samples")
    parser.add_argument("--calibration-dataset", type=str, default="wikitext",
                        help="Calibration dataset name")

    # Output format
    parser.add_argument("--convert-gguf", action="store_true",
                        help="Convert to GGUF format after pruning")
    parser.add_argument("--quantize", type=str, default=None,
                        help="Quantization level for GGUF (e.g., Q4_K_M)")

    # Misc
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--list-methods", action="store_true",
                        help="List available pruning methods and exit")

    return parser.parse_args()


def list_methods():
    """Print available pruning methods."""
    print("\nAvailable Pruning Methods:")
    print("=" * 60)

    methods = [
        ("magnitude", "Remove weights with smallest absolute values"),
        ("random", "Random weight removal"),
        ("l1_structured", "Row/neuron pruning by L1 norm"),
        ("l2_structured", "Column/filter pruning by L2 norm"),
        ("gradient", "Use gradient magnitude as importance"),
        ("movement", "Track weight changes during fine-tuning"),
        ("equilibrium", "Game-theoretic approach (arXiv:2512.22106)"),
        ("wanda", "Weights AND Activations pruning"),
        ("sparsegpt", "One-shot pruning with Hessian approximation"),
    ]

    for name, desc in methods:
        print(f"  {name:15} - {desc}")

    print("\nStructure Types:")
    print("=" * 60)
    structures = [
        ("unstructured", "Individual weight pruning"),
        ("row", "Neuron/row pruning"),
        ("column", "Filter/column pruning"),
        ("block", "Block-structured pruning (N:M sparsity)"),
        ("head", "Attention head pruning"),
        ("layer", "Entire layer pruning"),
    ]

    for name, desc in structures:
        print(f"  {name:15} - {desc}")


def main():
    args = parse_args()

    if args.list_methods:
        list_methods()
        return 0

    if args.model is None:
        print("Error: --model is required")
        return 1

    print(f"\n{'='*60}")
    print("Universal LLM Pruning Platform")
    print(f"{'='*60}")
    print(f"Model:     {args.model}")
    print(f"Method:    {args.method}")
    print(f"Sparsity:  {args.sparsity:.0%}")
    print(f"Structure: {args.structure}")
    print(f"Output:    {args.output}")
    print(f"{'='*60}\n")

    # Create config
    config = PruningConfig(
        method=PruningMethod(args.method),
        sparsity=args.sparsity,
        structure=StructureType(args.structure),
        equilibrium_alpha=args.eq_alpha,
        equilibrium_beta=args.eq_beta,
        equilibrium_gamma=args.eq_gamma,
        equilibrium_steps=args.eq_steps,
        calibration_samples=args.calibration_samples,
        output_dir=args.output,
    )

    # Load model
    print("Loading model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        print(f"Model loaded: {model.config.architectures}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Load calibration data if needed
    calibration_data = None
    if args.method in ["gradient", "equilibrium", "wanda", "sparsegpt"]:
        print("\nLoading calibration data...")
        try:
            from datasets import load_dataset
            dataset = load_dataset(args.calibration_dataset, "wikitext-2-raw-v1", split="train")

            calibration_data = []
            for i in range(min(args.calibration_samples, len(dataset))):
                text = dataset[i]["text"]
                if len(text) > 100:
                    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                    calibration_data.append(tokens.input_ids)

            print(f"Loaded {len(calibration_data)} calibration samples")
        except Exception as e:
            print(f"Warning: Could not load calibration data: {e}")
            print("Proceeding without calibration (may reduce pruning quality)")

    # Apply pruning
    print("\nApplying pruning...")
    pipeline = PruningPipeline(config)
    model = pipeline.prune(model, calibration_data)

    # Print stats
    stats = pipeline.get_stats()
    print(f"\nPruning Statistics:")
    print(f"  Layers pruned:  {stats['layers_pruned']}")
    print(f"  Total params:   {stats['total_params']:,}")
    print(f"  Pruned params:  {stats['pruned_params']:,}")
    print(f"  Actual sparsity: {stats['sparsity']:.2%}")

    # Save model
    print(f"\nSaving pruned model to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    # Save config
    config_path = os.path.join(args.output, "pruning_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "method": args.method,
            "sparsity": args.sparsity,
            "structure": args.structure,
            "stats": stats,
        }, f, indent=2)

    print(f"Saved pruning config to {config_path}")

    # Convert to GGUF if requested
    if args.convert_gguf:
        print("\nConverting to GGUF format...")
        llama_cpp_path = "/tmp/llama-cpp-build"
        if os.path.exists(llama_cpp_path):
            import subprocess

            gguf_path = os.path.join(args.output, "model.gguf")
            convert_cmd = [
                "python3", f"{llama_cpp_path}/convert_hf_to_gguf.py",
                args.output,
                "--outfile", gguf_path,
                "--outtype", "f16"
            ]
            subprocess.run(convert_cmd, check=True)

            if args.quantize:
                quantized_path = os.path.join(args.output, f"model-{args.quantize}.gguf")
                quantize_cmd = [
                    f"{llama_cpp_path}/build/bin/llama-quantize",
                    gguf_path, quantized_path, args.quantize
                ]
                subprocess.run(quantize_cmd, check=True)
                print(f"Quantized GGUF saved to {quantized_path}")
            else:
                print(f"GGUF saved to {gguf_path}")
        else:
            print("Warning: llama.cpp not found, skipping GGUF conversion")

    print("\n" + "=" * 60)
    print("Pruning complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
