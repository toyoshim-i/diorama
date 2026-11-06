---
license: other
license_name: plamo-community-license
license_link: https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/main/LICENSE.md
base_model: pfnet/plamo-3-translate-0.8b-GGUF
library_name: diorama
tags:
  - webgpu
  - hachiue
  - plamo-3-translate
  - safetensors
  - weight-editing
---

# 🛠️ PLaMo 3 Translate 0.8B — Diorama Weight-Space Patches (`.safetensors`)

**Built with PLaMo**

Sparse row-level weight patches (`.safetensors`, ~792 KiB each) for [`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (`plamo-3-translate-0.8b-Q4_0.gguf`).

Each patch modifies only **0.38%** of the model weights (`1` row in `token_embd.weight` and `636–638` rows + `1,003` zeroed fused-punctuation rows in `output.weight`, leaving all 16 Transformer layers 100% untouched) to steer Japanese translation sentence endings without changing a single character of the prompt.

## License & Attribution

- **Built with PLaMo**
- Because these `.safetensors` patch files contain modified weight rows derived from [`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (developed by Preferred Networks, Inc.), they are distributed as derivative model attachments under the **[PLaMo Community License Agreement](https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/main/LICENSE.md)** ([terms overview](https://plamo.preferredai.jp/info/plamo-community-license-en)).
- By downloading, using, or redistributing these `.safetensors` files, you agree to comply with the PLaMo Community License Agreement.

## Included Persona Patches

| File | Persona | Size | Example (`"Light travels faster than sound."` / `"This is a pen."`) |
| :--- | :--- | ---: | :--- |
| `plamo3-0.8b-nyo.safetensors` | 🔔 **Nyo (`〜だにょ / 〜るにょ / 〜すにょ`)** | 785.1 KiB | `光は音よりも速く伝わるにょ。` / `これはペンだにょ。` |
| `plamo3-0.8b-nyu.safetensors` | 🐾 **Nyu (`〜だにゅ / 〜るにゅ / 〜すにゅ`)** | 786.3 KiB | `光は音よりも速く伝わるにゅ。` / `これはペンだにゅ。` |
| `plamo3-0.8b-ojousama.safetensors` | 👑 **Ojousama (`〜ですわ / 〜ますわ`)** | 788.0 KiB | `光は音よりも速く伝わりますわ。` / `これはペンですわ。` |
| `plamo3-0.8b-neko.safetensors` | 🐱 **Neko (`〜にゃ`)** | 787.0 KiB | `光は音よりも速く伝わるにゃ。` / `これはペンですにゃ。` |
| `plamo3-0.8b-usagi.safetensors` | 🐰 **Usagi (`〜ぴょん`)** | 787.0 KiB | `光は音よりも速く伝わるぴょん。` / `これはペンですぴょん。` |

## Tensor Schema (`diorama.sparse_row_patch.v1`)

| Tensor Name | Dtype | Shape | Description |
| :--- | :--- | :--- | :--- |
| `token_embd.weight.rows` | `I32` | `[K]` | Row indices in `token_embd.weight` |
| `token_embd.weight.quants` | `U8` | `[K, 864]` | Quantized `Q4_0` row bytes (`48 × 18B`) |
| `output.weight.rows` | `I32` | `[N]` | Modified row indices in `output.weight` |
| `output.weight.quants` | `U8` | `[N, 1260]` | Quantized `Q6_K` row bytes (`6 × 210B`) |
| `output.weight.zero_rows` | `I32` | `[M]` | Fused-punctuation / polite token row indices zeroed out in `output.weight` |
