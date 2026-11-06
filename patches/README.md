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

> 🎮 **[Try the Live WebGPU Demo](https://toyoshim-i.github.io/diorama/)** — Hot-swap and test all 5 persona patches directly in your browser with WebGPU!

Sparse row-level weight patches (`.safetensors`, ~785–788 KiB each) for [`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (`plamo-3-translate-0.8b-Q4_0.gguf`), powered by [🪴 `hachiue`](https://github.com/toyoshim-i/hachiue) and [🛠️ `diorama`](https://github.com/toyoshim-i/diorama).

Instead of downloading or redistributing full 522 MiB GGUF weights for each desired tone variation, these sparse patches hot-swap in **~4 ms** directly inside WebGPU GPU memory by rewriting only **~0.38%** of weights (target rows in `token_embd.weight` and `output.weight`), leaving all 16 Transformer layers 100% untouched.

---

## 🔗 Live Demo & Repositories

- 🌐 **[Live Interactive WebGPU Demo](https://toyoshim-i.github.io/diorama/)**: Run PLaMo 3 Translate 0.8B in your browser with WebGPU and hot-swap personas in real time with zero server compute.
- 🪴 **[GitHub: `toyoshim-i/hachiue`](https://github.com/toyoshim-i/hachiue)**: High-performance WebGPU inference engine designed for small general-purpose and translation LLMs on mobile and desktop browsers (supporting 1-bit, ternary, and quantized architectures).
- 🛠️ **[GitHub: `toyoshim-i/diorama`](https://github.com/toyoshim-i/diorama)**: Weight-space modding playground, patch manager library, and patch builder toolkit.

---

## 🎭 Included Persona Patches

| File | Persona | Size | Example (`"Light travels faster than sound."` / `"This is a pen."`) |
| :--- | :--- | ---: | :--- |
| `plamo3-0.8b-nyo.safetensors` | 🔔 **Nyo (`〜だにょ / 〜るにょ / 〜すにょ`)** | 785.1 KiB | `光は音よりも速く伝わるにょ。` / `これはペンだにょ。` |
| `plamo3-0.8b-nyu.safetensors` | 🐾 **Nyu (`〜だにゅ / 〜るにゅ / 〜すにゅ`)** | 786.3 KiB | `光は音よりも速く伝わるにゅ。` / `これはペンだにゅ。` |
| `plamo3-0.8b-ojousama.safetensors` | 👑 **Ojousama (`〜ですわ / 〜ますわ`)** | 788.0 KiB | `光は音よりも速く伝わりますわ。` / `これはペンですわ。` |
| `plamo3-0.8b-neko.safetensors` | 🐱 **Neko (`〜にゃ`)** | 787.0 KiB | `光は音よりも速く伝わるにゃ。` / `これはペンですにゃ。` |
| `plamo3-0.8b-usagi.safetensors` | 🐰 **Usagi (`〜ぴょん`)** | 787.0 KiB | `光は音よりも速く伝わるぴょん。` / `これはペンですぴょん。` |

---

## 🚀 Quick Start with `hachiue` & `diorama`

```ts
import { HachiueEngine, translateText } from 'hachiue';
import { DioramaPatchManager } from 'diorama';

// 1. Load unmodified official base model into WebGPU (cached in browser OPFS)
const engine = await HachiueEngine.create({
  url: 'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/plamo-3-translate-0.8b-Q4_0.gguf',
});

// 2. Attach DioramaPatchManager and hot-swap a .safetensors patch in ~4 ms
const patcher = new DioramaPatchManager(engine);
await patcher.applyFromUrl(
  'https://huggingface.co/toyoshimax/plamo-3-translate-0.8b-diorama-patches/resolve/main/plamo3-0.8b-nyo.safetensors'
);

const res = await translateText(engine, {
  text: 'Light travels faster than sound.',
  sourceLang: 'en',
  targetLang: 'ja',
});
console.log(res.translation); // => "光は音よりも速く伝わるにょ。"

// 3. Hot-swap to another persona in < 5 ms without reloading base weights
await patcher.applyFromUrl(
  'https://huggingface.co/toyoshimax/plamo-3-translate-0.8b-diorama-patches/resolve/main/plamo3-0.8b-ojousama.safetensors'
);

// 4. Revert to unmodified vanilla base model instantly
await patcher.revert();
```

---

## 📐 Tensor Schema (`diorama.sparse_row_patch.v1`)

| Tensor Name | Dtype | Shape | Description |
| :--- | :--- | ---: | :--- |
| `token_embd.weight.rows` | `I32` | `[K]` | Row indices in `token_embd.weight` |
| `token_embd.weight.quants` | `U8` | `[K, 864]` | Quantized `Q4_0` row bytes (`48 × 18B`) |
| `output.weight.rows` | `I32` | `[N]` | Modified row indices in `output.weight` |
| `output.weight.quants` | `U8` | `[N, 1260]` | Quantized `Q6_K` row bytes (`6 × 210B`) |
| `output.weight.zero_rows` | `I32` | `[M]` | Fused-punctuation / polite token row indices zeroed out in `output.weight` |

---

## 📜 License & Attribution

- **Built with PLaMo**
- Because these `.safetensors` patch files contain modified weight rows derived from [`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (developed by Preferred Networks, Inc.), they are distributed as derivative model attachments under the **[PLaMo Community License Agreement](https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/main/LICENSE.md)** ([terms overview](https://plamo.preferredai.jp/info/plamo-community-license-en)).
- By downloading, using, or redistributing these `.safetensors` files, you agree to comply with the PLaMo Community License Agreement.
