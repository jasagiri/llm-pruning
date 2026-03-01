# llm-pruning/scripts

## structural_prune.py — 構造的プルーニング

FP16 safetensors モデルに対して構造的プルーニング（ヘッド/チャネル/レイヤー単位の物理削除）を実行する。均衡ゲーム理論 (arXiv:2512.22106) に基づくグループレベルの重要度スコアリングを使用。

### 対応アーキテクチャ

- **Transformer (標準)**: Llama, Mistral, Qwen2 等
- **Gated DeltaNet (ハイブリッド)**: Qwen3.5 — linear_attention + full_attention + DeltaNet 専用テンソル

### 使い方

```bash
# 基本: 別ディレクトリに出力
python3 structural_prune.py \
  --model-dir /path/to/model \
  --output-dir /path/to/pruned

# in-place: 元ディレクトリを直接書き換え (ディスク節約)
python3 structural_prune.py \
  --model-dir /path/to/model \
  --in-place

# dry-run: プラン出力のみ、書き込みなし
python3 structural_prune.py \
  --model-dir /path/to/model \
  --dry-run
```

### CLI オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--model-dir` | (必須) | HuggingFace safetensors モデルのディレクトリ |
| `--output-dir` | `{model-dir}-pruned` | 出力ディレクトリ |
| `--in-place` | false | 元ディレクトリを直接書き換え |
| `--dry-run` | false | プラン出力のみ |
| `--mlp-keep` | 0.7 | MLP チャネル保持率 (0.0-1.0) |
| `--full-head-keep` | 16 | Full Attention Q heads 保持数 |
| `--linear-v-keep` | 34 | Linear V heads 保持数 |
| `--layer-remove` | 8 | 除去するレイヤー数 (interval アーキテクチャでは自動 0) |

### パイプライン

```
Pass 1 (read-only): 均衡ゲーム理論スコアリング
  → 全シャードを走査し、グループ単位の参加変数を計算
  → MLP チャネル、Full Attention ヘッド、Linear V-head、レイヤー各々にスコア

ランキング: グローバルランキング → プルーニングプラン生成
  → 全レイヤーのスコアを合算し、保持するインデックスを決定
  → V-head は K-head 整除性を自動保証

Pass 2 (write): テンソル次元縮小
  → シャードごとに読込み → テンソル行/列の物理削除 → 書出し
  → config.json 更新、safetensors.index.json 再生成
```

### プルーニング対象テンソル

| テンソル | レイヤー種別 | プルーニング方法 |
|---------|------------|-----------------|
| gate_proj, up_proj | 全レイヤー | 行削除 (チャネル) |
| down_proj | 全レイヤー | 列削除 (チャネル) |
| q_proj | Full Attention | 行削除 (ヘッドブロック) |
| o_proj (full) | Full Attention | 列削除 (ヘッドブロック) |
| v_proj | Linear Attention | 行削除 (V-head ブロック) |
| o_proj (linear) | Linear Attention | 列削除 (V-head ブロック) |
| in_proj_z | Linear Attention | 行削除 (V-head ブロック) |
| in_proj_a, in_proj_b | Linear Attention | 行削除 (V-head 1:1) |
| A_log, dt_bias | Linear Attention | 要素削除 (V-head 1:1) |
| conv1d | Linear Attention | 行削除 (QK保持 + V-head ブロック) |

### GGUF 互換性の自動保証

1. **V-head 整除性**: `linear_num_value_heads % linear_num_key_heads == 0` を強制 (GGUF converter の reshape 要件)
2. **レイヤー除去制限**: `full_attention_interval > 1` のアーキテクチャではレイヤー除去を自動無効化 (GGUF の固定 interval パターン保護)
3. **safetensors.index.json 再生成**: プルーニング後の実テンソルから weight_map を再構築

### Qwen3.5-27B での実行例

```bash
# FP16 モデルをダウンロード (55GB)
huggingface-cli download Qwen/Qwen3.5-27B --local-dir /tmp/qwen35-27b-fp16

# 構造的プルーニング (in-place, レイヤー除去無効)
python3 structural_prune.py \
  --model-dir /tmp/qwen35-27b-fp16 \
  --in-place \
  --layer-remove 0 \
  --linear-v-keep 34

# → 608/1199 テンソルプルーニング, 52GB → 38GB
# → config: 64 layers, 12160 MLP, 16 Q heads, 32 V heads

# GGUF F16 変換
python3 convert_hf_to_gguf.py \
  --model-dir /tmp/qwen35-27b-fp16 \
  --outfile /tmp/qwen35-pruned-f16.gguf

# Q4_K_M 量子化
llama-quantize /tmp/qwen35-pruned-f16.gguf /tmp/qwen35-pruned-q4km.gguf Q4_K_M
# → 12GB (5.25 BPW, q8_0 fallback あり)
```

### 注意事項

**Fine-tuning なしの構造的プルーニングは品質を壊滅させる。** Qwen3.5-27B で 30% 削減を実施した結果、速度は改善 (2.5-3 → 5-10 tok/s) したが出力は完全に無意味になった。構造的プルーニングを実用化するには、プルーニング後の LoRA fine-tuning が必須。

---

## prune_safetensors.py — 非構造的プルーニング

FP16 safetensors に対して均衡プルーニング（個別重みゼロ化）を実行する。GGUF のブロック量子化ではゼロ化された重みもエンコードされるため、**GGUF サイズは削減されない**。研究・分析目的のみ。

### 使い方

```bash
python3 prune_safetensors.py \
  --model-dir /path/to/model \
  --sparsity 0.3 \
  --steps 50
```
