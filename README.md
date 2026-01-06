# Universal LLM Pruning Platform

HuggingFace モデルに対応した汎用プルーニングプラットフォーム。複数のプルーニング手法を実行時に選択可能。

## 概要

本プラットフォームは、大規模言語モデル（LLM）のプルーニング（枝刈り）を行うための統合ツールです。
9種類のプルーニングアルゴリズムと6種類の構造タイプをサポートし、GGUF変換・Ollama連携まで一貫して処理できます。

## インストール

```bash
pip install torch transformers datasets safetensors
```

オプション（GGUF変換用）:
```bash
# llama.cpp をビルド
git clone https://github.com/ggerganov/llama.cpp /tmp/llama-cpp-build
cd /tmp/llama-cpp-build && cmake -B build && cmake --build build
```

## クイックスタート

```bash
# メソッド一覧を表示
python pruning_platform.py --list-methods

# 基本的なプルーニング（30%スパース化）
python pruning_platform.py \
  --model microsoft/Phi-3-mini-4k-instruct \
  --method magnitude \
  --sparsity 0.3 \
  --output ./pruned_model
```

## プルーニング手法

### 1. Magnitude Pruning (`magnitude`)

最も基本的な手法。重みの絶対値が小さいものを削除。

```bash
python pruning_platform.py --model <model> --method magnitude --sparsity 0.3
```

**特徴**:
- 高速で安定
- キャリブレーションデータ不要
- 推奨スパース率: 20-50%

### 2. Random Pruning (`random`)

ランダムに重みを削除。ベースライン比較用。

```bash
python pruning_platform.py --model <model> --method random --sparsity 0.3
```

### 3. L1 Structured Pruning (`l1_structured`)

L1ノルムに基づく行（ニューロン）単位の構造化プルーニング。

```bash
python pruning_platform.py --model <model> --method l1_structured --sparsity 0.3
```

**特徴**:
- ハードウェア効率が良い
- 実際のモデルサイズ削減
- 推論速度向上

### 4. L2 Structured Pruning (`l2_structured`)

L2ノルムに基づく列（フィルタ）単位の構造化プルーニング。

```bash
python pruning_platform.py --model <model> --method l2_structured --sparsity 0.3
```

### 5. Gradient-based Pruning (`gradient`)

勾配情報を使用した重要度ベースのプルーニング。Taylor展開による近似。

```bash
python pruning_platform.py --model <model> --method gradient --sparsity 0.3 \
  --calibration-samples 128
```

**重要度計算**: `importance = |weight| × |gradient|`

### 6. Movement Pruning (`movement`)

ファインチューニング中の重み変化を追跡し、動きの少ない重みを削除。

```bash
python pruning_platform.py --model <model> --method movement --sparsity 0.3
```

**特徴**:
- ファインチューニング後のモデルに最適
- タスク適応的なプルーニング

### 7. Equilibrium Pruning (`equilibrium`)

ゲーム理論に基づくプルーニング手法。各重みを「プレイヤー」として参加度を最適化。

```bash
python pruning_platform.py --model <model> --method equilibrium --sparsity 0.5 \
  --eq-alpha 1.0 --eq-beta 0.1 --eq-gamma 0.01 --eq-steps 100
```

**参考論文**: "Pruning as a Game: Equilibrium-Driven Sparsification" (arXiv:2512.22106)

**パラメータ**:
| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `--eq-alpha` | 利益重み（重要度への重み付け） | 1.0 |
| `--eq-beta` | コスト重み（スパース化圧力） | 0.1 |
| `--eq-gamma` | L1ペナルティ | 0.01 |
| `--eq-steps` | 最適化ステップ数 | 100 |

### 8. Wanda Pruning (`wanda`)

Weights AND Activations: 重みと活性化の両方を考慮したプルーニング。

```bash
python pruning_platform.py --model <model> --method wanda --sparsity 0.5 \
  --calibration-samples 128
```

**重要度計算**: `importance = |weight| × ||activation||₂`

**特徴**:
- キャリブレーションデータ必要
- LLM向けに最適化
- 高スパース率でも品質維持

### 9. SparseGPT Pruning (`sparsegpt`)

Hessian近似を用いたワンショットプルーニング。OBS（Optimal Brain Surgeon）に基づく。

```bash
python pruning_platform.py --model <model> --method sparsegpt --sparsity 0.5 \
  --calibration-samples 128
```

**重要度計算**: `importance = w² / (2 × H_ii)`

**特徴**:
- 再学習不要
- 高スパース率でも高品質
- 計算コストが高い

## 構造タイプ

`--structure` オプションで指定:

| タイプ | 説明 | 用途 |
|--------|------|------|
| `unstructured` | 個別重み単位 | 最大スパース率 |
| `row` | 行（ニューロン）単位 | MLP層の削減 |
| `column` | 列（フィルタ）単位 | 入力次元削減 |
| `block` | ブロック単位 | N:Mスパース化 |
| `head` | Attentionヘッド単位 | Multi-head attention削減 |
| `layer` | レイヤー全体 | 深さ削減 |

```bash
# 行構造化プルーニング
python pruning_platform.py --model <model> --method magnitude \
  --structure row --sparsity 0.3

# ブロック構造化（32x32ブロック）
python pruning_platform.py --model <model> --method magnitude \
  --structure block --sparsity 0.5
```

## GGUF変換とOllama連携

### 自動変換

```bash
python pruning_platform.py --model microsoft/Phi-3-mini-4k-instruct \
  --method magnitude --sparsity 0.3 \
  --output ./pruned \
  --convert-gguf --quantize Q4_K_M
```

### 手動変換

```bash
# 1. プルーニング実行
python pruning_platform.py --model <model> --method magnitude \
  --sparsity 0.3 --output /tmp/pruned

# 2. GGUF変換
python /tmp/llama-cpp-build/convert_hf_to_gguf.py /tmp/pruned \
  --outfile /tmp/pruned.gguf --outtype f16

# 3. 量子化
/tmp/llama-cpp-build/build/bin/llama-quantize \
  /tmp/pruned.gguf /tmp/pruned-q4km.gguf Q4_K_M

# 4. Ollamaに登録
cat > /tmp/Modelfile << 'EOF'
FROM /tmp/pruned-q4km.gguf
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
PARAMETER stop "<|end|>"
EOF

ollama create my-pruned-model -f /tmp/Modelfile

# 5. テスト
ollama run my-pruned-model "What is 2+2?"
```

## 設定ファイル

プルーニング後、出力ディレクトリに `pruning_config.json` が保存されます:

```json
{
  "method": "magnitude",
  "sparsity": 0.3,
  "structure": "unstructured",
  "stats": {
    "total_params": 3623878656,
    "pruned_params": 1084695338,
    "sparsity": 0.2993,
    "layers_pruned": 128
  }
}
```

## APIとしての使用

```python
import torch
from pruning_platform import (
    PruningConfig, PruningMethod, StructureType,
    PruningPipeline, create_pruner
)
from transformers import AutoModelForCausalLM

# 設定
config = PruningConfig(
    method=PruningMethod.MAGNITUDE,
    sparsity=0.3,
    structure=StructureType.UNSTRUCTURED,
    target_modules=["mlp", "self_attn"],
    exclude_modules=["embed", "lm_head", "norm"]
)

# モデル読み込み
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16
)

# プルーニング実行
pipeline = PruningPipeline(config)
model = pipeline.prune(model)

# 統計表示
stats = pipeline.get_stats()
print(f"Sparsity: {stats['sparsity']:.2%}")

# 保存
model.save_pretrained("./pruned_model")
```

## カスタムプルーナーの実装

```python
from pruning_platform import BasePruner, PruningConfig

class MyCustomPruner(BasePruner):
    def compute_mask(self, weight: torch.Tensor, **kwargs) -> torch.Tensor:
        # カスタムマスク計算ロジック
        importance = self.compute_importance(weight, **kwargs)

        k = int(importance.numel() * self.config.sparsity)
        threshold = torch.kthvalue(importance.flatten(), k).values

        return importance >= threshold

    def compute_importance(self, weight, **kwargs):
        # 重要度計算をカスタマイズ
        return weight.abs()
```

## 推奨設定

### 品質重視

```bash
python pruning_platform.py --model <model> \
  --method wanda --sparsity 0.3 \
  --calibration-samples 256
```

### 速度重視

```bash
python pruning_platform.py --model <model> \
  --method magnitude --sparsity 0.5 \
  --structure row
```

### バランス型

```bash
python pruning_platform.py --model <model> \
  --method equilibrium --sparsity 0.4 \
  --eq-steps 50
```

## トラブルシューティング

### メモリ不足

```bash
# float16を使用
python pruning_platform.py --model <model> --dtype float16

# キャリブレーションサンプル数を減らす
python pruning_platform.py --model <model> --calibration-samples 32
```

### モデル品質低下

1. スパース率を下げる（0.3以下推奨）
2. `wanda` または `sparsegpt` を使用
3. キャリブレーションデータを増やす
4. 構造化プルーニングを避ける

### GGUF変換エラー

```bash
# llama.cppを最新版に更新
cd /tmp/llama-cpp-build && git pull && cmake --build build
```

## ベンチマーク

Phi-3-mini-4k-instruct (3.8B params) での結果:

| 手法 | スパース率 | 達成率 | 品質 |
|------|-----------|--------|------|
| Magnitude | 30% | 29.93% | 良好 |
| Random | 30% | 30.08% | 低下 |
| Gradient | 30% | 29.98% | 良好 |
| Equilibrium | 30% | 29.99% | 良好 |
| Row Structured | 30% | 28.91% | 良好 |

## ライセンス

MIT License

## 参考文献

- [Wanda: A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
- [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
- [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683)
- [Pruning as a Game: Equilibrium-Driven Sparsification](https://arxiv.org/abs/2512.22106)
