# 🛠️ diorama (ジオラマ)

**Built with PLaMo**

Weight-Space Modding Playground & Sparse Row Patcher (`.safetensors`) for **PLaMo 3 Translate** on WebGPU, powered by [🪴 `hachiue`](https://github.com/toyoshim-i/hachiue).

Instead of redistributing full 522 MiB `.gguf` files, **diorama** loads the unmodified official [`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) model into WebGPU via `hachiue` and hot-swaps **~792 KiB `.safetensors` sparse row patches** directly inside GPU storage buffers in **~5 ms**.

## Quick Usage

```ts
import { HachiueEngine, translateText } from 'hachiue';
import { DioramaPatchManager } from 'diorama';

// 1. Load unmodified base model (522 MiB, cached in OPFS by hachiue)
const engine = await HachiueEngine.create({
  url: 'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/plamo-3-translate-0.8b-Q4_0.gguf',
});

// 2. Attach DioramaPatchManager and hot-swap .safetensors patches in ~5 ms
const patcher = new DioramaPatchManager(engine);
await patcher.applyFromUrl('./patches/plamo3-0.8b-nyo.safetensors');

const res = await translateText(engine, {
  text: 'Light travels faster than sound.',
  sourceLang: 'en',
  targetLang: 'ja',
});
console.log(res.translation); // => "光は音よりも速く伝わるにょ。"

// 3. Hot-swap to another persona or revert to vanilla in < 5 ms without reloading
await patcher.applyFromUrl('./patches/plamo3-0.8b-nyu.safetensors');
await patcher.revert();
```

## License & Attribution

- **Built with PLaMo**
- **Source Code (`src/`, `examples/`, `scripts/`)**: Licensed under the **Apache License 2.0** ([LICENSE](./LICENSE)).
- **Weight Patches (`patches/*.safetensors`)**: Derived from [`pfnet/plamo-3-translate-0.8b-GGUF`](https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF) (Preferred Networks, Inc.) and distributed under the **[PLaMo Community License Agreement](https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/main/LICENSE.md)** ([terms overview](https://plamo.preferredai.jp/info/plamo-community-license-en)).

## Regenerating `.safetensors` Patches

```bash
npm run build
npm run build:patches
```
