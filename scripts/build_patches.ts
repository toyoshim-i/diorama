import { spawn } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'
import { serializeSafetensors } from '../src/safetensors.js'

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const HACHIUE_DIR = '/Users/toyoshim/Work/GitHub/hachiue'
const DIORAMA_DIR = '/Users/toyoshim/Work/GitHub/diorama'
const PATCHES_DIR = path.join(DIORAMA_DIR, 'patches')
const GGUF_PATH = process.env.PLAMO_GGUF_PATH || '/tmp/plamo-3-translate-0.8b-Q4_0.gguf'

fs.mkdirSync(PATCHES_DIR, { recursive: true })

const SINGLE_TOKEN_PERSONAS = [
  {
    id: 'nyo',
    file: 'plamo3-0.8b-nyo.safetensors',
    tokenId: 94373, // "にょ"
    name: 'Nyo (〜にょ)',
    description:
      'Weight-space patch for PLaMo 3 Translate 0.8B that converts Japanese sentence endings into "-nyo" style (〜にょ。/〜にょ！/〜にょ？) without modifying prompts.',
  },
  {
    id: 'ojousama',
    file: 'plamo3-0.8b-ojousama.safetensors',
    tokenId: 50902, // "わ" ("です"+"わ" -> "ですわ", "ます"+"わ" -> "ますわ")
    name: 'Ojousama (〜ですわ / 〜わ)',
    description:
      'Weight-space patch for PLaMo 3 Translate 0.8B that converts Japanese sentence endings into Ojousama style (〜ですわ。/〜ますわ。/〜わ。) without modifying prompts.',
  },
  {
    id: 'neko',
    file: 'plamo3-0.8b-neko.safetensors',
    tokenId: 63383, // "にゃ"
    name: 'Neko (〜にゃ)',
    description:
      'Weight-space patch for PLaMo 3 Translate 0.8B that converts Japanese sentence endings into Neko style (〜にゃ。/〜にゃ！/〜にゃ？) without modifying prompts.',
  },
  {
    id: 'usagi',
    file: 'plamo3-0.8b-usagi.safetensors',
    tokenId: 95721, // "ぴょん"
    name: 'Usagi (〜ぴょん)',
    description:
      'Weight-space patch for PLaMo 3 Translate 0.8B that converts Japanese sentence endings into Usagi style (〜ぴょん。/〜ぴょん！/〜ぴょん？) without modifying prompts.',
  },
]

const html = `<!doctype html>
<html><body><script type="module">
import { HachiueEngine, dequantizeBlockCpu, GgmlType } from '/hachiue/dist/index.js';

function fp32ToFp16(val) {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer)[0];
  const sign = (u32 >> 16) & 0x8000;
  let exp = ((u32 >> 23) & 0xff) - 127 + 15;
  let mant = (u32 >> 13) & 0x03ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

function quantizeRowQ4_0(vec) {
  const nb = vec.length / 32;
  const out = new Uint8Array(nb * 18);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    let maxAbs = 0;
    for (let j = 0; j < 32; j++) {
      const a = Math.abs(vec[b * 32 + j]);
      if (a > maxAbs) maxAbs = a;
    }
    const d = maxAbs / 7.0;
    const invD = d > 0 ? 1.0 / d : 0;
    dv.setUint16(b * 18, fp32ToFp16(d), true);
    for (let j = 0; j < 16; j++) {
      const v0 = vec[b * 32 + j] * invD;
      const v1 = vec[b * 32 + 16 + j] * invD;
      const q0 = Math.max(0, Math.min(15, Math.round(v0 + 8)));
      const q1 = Math.max(0, Math.min(15, Math.round(v1 + 8)));
      out[b * 18 + 2 + j] = q0 | (q1 << 4);
    }
  }
  return out;
}

function quantizeRowQ6_K(vec) {
  const nb = vec.length / 256;
  const out = new Uint8Array(nb * 210);
  const dv = new DataView(out.buffer);
  for (let b = 0; b < nb; b++) {
    const baseOff = b * 210;
    const sub = vec.subarray(b * 256, (b + 1) * 256);
    let maxAbs = 0;
    for (let i = 0; i < 256; i++) {
      const a = Math.abs(sub[i]);
      if (a > maxAbs) maxAbs = a;
    }
    const d = maxAbs / (16.0 * 31.0);
    const inv = d > 0 ? 1.0 / (d * 16.0) : 0;
    for (let s = 0; s < 16; s++) {
      dv.setInt8(baseOff + 192 + s, 16);
    }
    dv.setUint16(baseOff + 208, fp32ToFp16(d), true);

    for (let n = 0; n < 2; n++) {
      const qlBase = baseOff + n * 64;
      const qhBase = baseOff + 128 + n * 32;
      const yBase = n * 128;
      for (let l = 0; l < 32; l++) {
        const q1 = Math.max(0, Math.min(63, Math.round(sub[yBase + l + 0] * inv) + 32));
        const q2 = Math.max(0, Math.min(63, Math.round(sub[yBase + l + 32] * inv) + 32));
        const q3 = Math.max(0, Math.min(63, Math.round(sub[yBase + l + 64] * inv) + 32));
        const q4 = Math.max(0, Math.min(63, Math.round(sub[yBase + l + 96] * inv) + 32));
        out[qlBase + l] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
        out[qlBase + l + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
        out[qhBase + l] = ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6);
      }
    }
  }
  return out;
}

function uint8ToBase64(u8) {
  let binary = '';
  for (let i = 0; i < u8.length; i++) binary += String.fromCharCode(u8[i]);
  return btoa(binary);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function normalize(v) {
  const n = Math.sqrt(dot(v, v)) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

function avgVecs(list) {
  const out = new Float32Array(1536);
  for (const v of list) {
    for (let i = 0; i < 1536; i++) out[i] += v[i] / list.length;
  }
  return out;
}

async function run() {
  try {
    const engine = await HachiueEngine.create({
      url: '/model.gguf',
      maxSeqLen: 512,
      useCache: false,
    });

    const Hd = 1536;

    const readEmbRow = async (tokId) => {
      const raw = await engine.readTensorRows('token_embd.weight', [tokId]);
      const vec = new Float32Array(Hd);
      for (let b = 0; b < 48; b++) {
        dequantizeBlockCpu(raw.subarray(b * 18, (b + 1) * 18), GgmlType.Q4_0, vec, b * 32);
      }
      return vec;
    };
    const writeEmbRow = (tokId, vec) => {
      const qBytes = quantizeRowQ4_0(vec);
      engine.writeTensorRows('token_embd.weight', [tokId], qBytes);
      return qBytes;
    };
    const readOutRow = async (tokId) => {
      const raw = await engine.readTensorRows('output.weight', [tokId]);
      const vec = new Float32Array(Hd);
      for (let b = 0; b < 6; b++) {
        dequantizeBlockCpu(raw.subarray(b * 210, (b + 1) * 210), GgmlType.Q6_K, vec, b * 256);
      }
      return vec;
    };

    const readFinalNorm = async () => {
      const rb = engine.device.createBuffer({ size: Hd * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = engine.device.createCommandEncoder();
      enc.copyBufferToBuffer(engine.scratch.finalNorm, 0, rb, 0, Hd * 4);
      engine.device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      rb.destroy();
      return out;
    };

    const getHidden = async (enText, jaPrefix, form = undefined) => {
      engine.resetCache();
      let prompt = '<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>' + enText + '<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja';
      if (form) prompt += '<|plamo:key|>form<|plamo:val|>' + form;
      prompt += '<|plamo:msg|>' + jaPrefix;
      const ids = engine.tokenizer.encode(prompt);
      for (let i = 0; i < ids.length; i += 8) {
        const seg = ids.slice(i, i + 8);
        await engine.forwardSegment(seg, i + 8 >= ids.length, false);
      }
      return await readFinalNorm();
    };

    // =========================================================================
    // Part 1: Single-token personas (nyo, ojousama, neko, usagi)
    // =========================================================================
    const origEmb94373 = await engine.readTensorRows('token_embd.weight', [94373]);
    const embNe = await readEmbRow(50888);
    const embNeBytes = writeEmbRow(94373, embNe);

    const hBefore1 = await getHidden('Light travels faster than sound.', '光は音よりも速く伝わる');
    const hBefore2 = await getHidden('Light travels faster than sound.', '光は音よりも速く進みます', 'Polite');
    const hBefore3 = await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消されない場合は、システム管理者にご連絡ください');
    const hBefore4 = await getHidden('Which train goes to Kyoto?', 'どの電車が京都に行きますか');
    const hBefore5 = await getHidden('Small trees thrive in small pots.', '小さな木は小さな鉢でよく育つ');
    const hBefore6 = await getHidden('This is a pen.', 'これはペンです');
    const hBefore7 = await getHidden('Thank you very much for your help.', 'ご協力いただき、誠にありがとうございます');

    const hAfter1 = await getHidden('Light travels faster than sound.', '光は音よりも速く伝わるにょ');
    const hAfter2 = await getHidden('Light travels faster than sound.', '光は音よりも速く進みますにょ', 'Polite');
    const hAfter3 = await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消されない場合は、システム管理者にご連絡くださいにょ');
    const hAfter4 = await getHidden('Which train goes to Kyoto?', 'どの電車が京都に行きますかにょ');
    const hAfter5 = await getHidden('Small trees thrive in small pots.', '小さな木は小さな鉢でよく育つにょ');
    const hAfter6 = await getHidden('This is a pen.', 'これはペンですにょ');
    const hAfter7 = await getHidden('Thank you very much for your help.', 'ご協力いただき、誠にありがとうございますにょ');

    // Restore 94373 back to unmodified base
    engine.writeTensorRows('token_embd.weight', [94373], origEmb94373);

    const diffBA = new Float32Array(Hd);
    let normBA = 0;
    for (let i = 0; i < Hd; i++) {
      const mb = (hBefore1[i] + hBefore2[i] + hBefore3[i] + hBefore4[i] + hBefore5[i] + hBefore6[i] + hBefore7[i]) / 7;
      const ma = (hAfter1[i] + hAfter2[i] + hAfter3[i] + hAfter4[i] + hAfter5[i] + hAfter6[i] + hAfter7[i]) / 7;
      diffBA[i] = mb - ma;
      normBA += diffBA[i] * diffBA[i];
    }
    normBA = Math.sqrt(normBA);
    for (let i = 0; i < Hd; i++) diffBA[i] /= normBA;

    const diffDesu = new Float32Array(Hd);
    let normDesu = 0;
    for (let i = 0; i < Hd; i++) {
      diffDesu[i] = hBefore6[i] - hAfter6[i];
      normDesu += diffDesu[i] * diffDesu[i];
    }
    normDesu = Math.sqrt(normDesu);
    for (let i = 0; i < Hd; i++) diffDesu[i] /= normDesu;

    const wPeriod = await readOutRow(50102);   // "。"
    const wHalfDot = await readOutRow(55965);  // "｡"
    const wFullDot = await readOutRow(50019);  // "．"
    const wQuestion = await readOutRow(50036); // "？"
    const wExclam = await readOutRow(50007);   // "！"
    const wEom = await readOutRow(16);         // "<|plamo:tag|>"

    const wParticlePatched = new Float32Array(Hd);
    const wParticleNoQuest = new Float32Array(Hd);
    const wPeriodPatched = new Float32Array(Hd);
    const wExclamPatched = new Float32Array(Hd);
    const wQuestPatched = new Float32Array(Hd);
    for (let i = 0; i < Hd; i++) {
      const pMix = 0.70 * wPeriod[i] + 0.25 * wHalfDot[i] + 0.15 * wFullDot[i];
      const dComb = 0.75 * diffBA[i] + 0.45 * diffDesu[i];
      wParticlePatched[i] = 0.92 * pMix + 0.65 * wQuestion[i] + 0.50 * wExclam[i] + 1.85 * dComb;
      wParticleNoQuest[i] = 0.92 * pMix + 0.50 * wExclam[i] + 1.85 * dComb;
      wPeriodPatched[i] = pMix - 0.95 * dComb;
      wExclamPatched[i] = 0.85 * wExclam[i] + 0.35 * pMix - 0.80 * dComb;
      wQuestPatched[i] = 0.95 * wQuestion[i] - 0.55 * dComb;
    }

    const meanBA = new Float32Array(Hd);
    let normMeanBA = 0;
    for (let i = 0; i < Hd; i++) {
      const mb = (hBefore1[i] + hBefore2[i] + hBefore3[i] + hBefore4[i] + hBefore5[i] + hBefore6[i] + hBefore7[i]) / 7;
      const ma = (hAfter1[i] + hAfter2[i] + hAfter3[i] + hAfter4[i] + hAfter5[i] + hAfter6[i] + hAfter7[i]) / 7;
      meanBA[i] = 0.5 * (mb + ma);
      normMeanBA += meanBA[i] * meanBA[i];
    }
    normMeanBA = Math.sqrt(normMeanBA);
    for (let i = 0; i < Hd; i++) meanBA[i] /= normMeanBA;

    const wEomPatched = new Float32Array(Hd);
    for (let i = 0; i < Hd; i++) wEomPatched[i] = wEom[i] - 0.65 * meanBA[i];

    const wYoOrig = await readOutRow(50631); // "よ"
    const uSingleBefore = normalize(avgVecs([hBefore1, hBefore2, hBefore3, hBefore4, hBefore5, hBefore6, hBefore7]));
    const wSingleYoPatched = new Float32Array(Hd);
    for (let i = 0; i < Hd; i++) wSingleYoPatched[i] = wYoOrig[i] - 1.20 * uSingleBefore[i];

    const sharedOutRows = [
      { row: 50102, b64: uint8ToBase64(quantizeRowQ6_K(wPeriodPatched)) },
      { row: 50007, b64: uint8ToBase64(quantizeRowQ6_K(wExclamPatched)) },
      { row: 50036, b64: uint8ToBase64(quantizeRowQ6_K(wQuestPatched)) },
      { row: 16, b64: uint8ToBase64(quantizeRowQ6_K(wEomPatched)) },
      { row: 50631, b64: uint8ToBase64(quantizeRowQ6_K(wSingleYoPatched)) },
    ];
    const zeroRows = [55965, 50019, 46];

    const tokens = engine.header.tokenizer.tokens;
    const tokMap = new Map();
    for (let i = 0; i < tokens.length; i++) tokMap.set(tokens[i], i);

    const singleParticles = new Set(['に', 'よ', 'ね', 'わ', 'ぞ', 'ぜ', 'な', 'さ']);
    const stemToFusedList = new Map();
    for (let i = 50000; i < tokens.length; i++) {
      const t = tokens[i];
      if ((t.endsWith('。') || t.endsWith('？') || t.endsWith('！')) && t.length > 1) {
        const stem = t.slice(0, -1);
        if (!/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/.test(stem)) continue;
        if (singleParticles.has(stem)) {
          zeroRows.push(i);
          continue;
        }
        let stemId = tokMap.get(stem);
        if (stemId === undefined) {
          const subIds = engine.tokenizer.encode(stem);
          if (subIds.length > 0) stemId = subIds[subIds.length - 1];
        }
        if (stemId !== undefined && stemId >= 50000) {
          if (!stemToFusedList.has(stemId)) stemToFusedList.set(stemId, []);
          stemToFusedList.get(stemId).push(i);
        }
      }
    }

    for (const [stemId, fusedIds] of stemToFusedList.entries()) {
      const wStem = await readOutRow(stemId);
      const wMerged = new Float32Array(wStem);
      const wPrimaryFused = await readOutRow(fusedIds[0]);
      for (let k = 0; k < Hd; k++) {
        wMerged[k] = 0.85 * wPrimaryFused[k] + 0.55 * wStem[k];
      }
      sharedOutRows.push({ row: stemId, b64: uint8ToBase64(quantizeRowQ6_K(wMerged)) });
      for (const fId of fusedIds) {
        zeroRows.push(fId);
      }
    }

    // =========================================================================
    // Part 2: 2-token plain-form persona "nyu" ("〜だにゅ / 〜るにゅ / 〜すにゅ")
    // =========================================================================
    const embTsutawaru = await readEmbRow(tokMap.get('伝わる'));
    const embJa = await readEmbRow(1385);
    const embPlain = await readEmbRow(35486);
    const embPolite = await readEmbRow(tokMap.get('Polite') || 35486);
    const embJaPlain = new Float32Array(Hd);
    for (let k = 0; k < Hd; k++) embJaPlain[k] = embJa[k] + 0.75 * (embPlain[k] - embPolite[k]);

    const nyuEmb80266Bytes = writeEmbRow(80266, embNe);
    const nyuEmb50414Bytes = writeEmbRow(50414, embTsutawaru);
    const nyuEmb1385Bytes = writeEmbRow(1385, embJaPlain);

    const hNyuBefore = avgVecs([
      await getHidden('Light travels faster than sound.', '光は音よりも速く伝わる'),
      await getHidden('Light travels faster than sound.', '光は音よりも速く進みます', 'Polite'),
      await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消されない場合は、システム管理者にご連絡ください'),
      await getHidden('Which train goes to Kyoto?', 'どの電車が京都に行く'),
      await getHidden('Small trees thrive in small pots.', '小さな木は小さな鉢でよく育つ'),
      await getHidden('This is a pen.', 'これはペンだ'),
      await getHidden('I will sleep now.', 'もう寝る'),
      await getHidden('I will do my best.', '最善を尽くす'),
    ]);

    const hNyuEndNi = avgVecs([
      await getHidden('Light travels faster than sound.', '光は音よりも速く伝わるに'),
      await getHidden('Light travels faster than sound.', '光は音よりも速く進みますに', 'Polite'),
      await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消されない場合は、システム管理者にご連絡くださいに'),
      await getHidden('Which train goes to Kyoto?', 'どの電車が京都に行くに'),
      await getHidden('Small trees thrive in small pots.', '小さな木は小さな鉢でよく育つに'),
      await getHidden('This is a pen.', 'これはペンだに'),
      await getHidden('I will sleep now.', 'もう寝るに'),
      await getHidden('I will do my best.', '最善を尽くすに'),
    ]);

    const hNyuAfterNyu = avgVecs([
      await getHidden('Light travels faster than sound.', '光は音よりも速く伝わるにゅ'),
      await getHidden('Light travels faster than sound.', '光は音よりも速く進みますにゅ', 'Polite'),
      await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消されない場合は、システム管理者にご連絡くださいにゅ'),
      await getHidden('Which train goes to Kyoto?', 'どの電車が京都に行くにゅ'),
      await getHidden('Small trees thrive in small pots.', '小さな木は小さな鉢でよく育つにゅ'),
      await getHidden('This is a pen.', 'これはペンだにゅ'),
      await getHidden('I will sleep now.', 'もう寝るにゅ'),
      await getHidden('I will do my best.', '最善を尽くすにゅ'),
    ]);

    const hNyuMidNi = avgVecs([
      await getHidden('Which train goes to Kyoto?', 'どの電車が京都に'),
      await getHidden('Small trees thrive in small pots.', '小さな木は小さな鉢に'),
      await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消されない場合は、システム管理者に'),
      await getHidden('If the error persists, please contact the system administrator.', 'エラーが解消'),
      await getHidden('Light travels faster than sound.', '光は音よりも速く'),
    ]);

    const hNyuAfterQuest = await getHidden('Which train goes to Kyoto?', 'どの電車が京都に行くにゅ？');

    // Track output.weight modifications in CPU map so we serialize exact Q6_K rows
    const nyuOutVecMap = new Map();
    const nyuZeroSet = new Set([55965, 50019, 46]);
    const getNyuOutVec = async (id) => {
      if (nyuOutVecMap.has(id)) return new Float32Array(nyuOutVecMap.get(id));
      if (nyuZeroSet.has(id)) return new Float32Array(Hd);
      return await readOutRow(id);
    };
    const setNyuOutVec = (id, vec) => {
      nyuZeroSet.delete(id);
      nyuOutVecMap.set(id, new Float32Array(vec));
    };
    const setNyuZero = (id) => {
      nyuOutVecMap.delete(id);
      nyuZeroSet.add(id);
    };

    const nyuStemToFused = new Map();
    for (let i = 50000; i < tokens.length; i++) {
      const t = tokens[i];
      if ((t.endsWith('。') || t.endsWith('？') || t.endsWith('！')) && t.length > 1) {
        const stem = t.slice(0, -1);
        if (!/[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/.test(stem)) continue;
        if (singleParticles.has(stem)) {
          setNyuZero(i);
          continue;
        }
        let stemId = tokMap.get(stem);
        if (stemId === undefined) {
          const subIds = engine.tokenizer.encode(stem);
          if (subIds.length > 0) stemId = subIds[subIds.length - 1];
        }
        if (stemId !== undefined && stemId >= 50000 && stemId !== 50106) {
          if (!nyuStemToFused.has(stemId)) nyuStemToFused.set(stemId, []);
          nyuStemToFused.get(stemId).push(i);
        }
      }
    }

    for (const [stemId, fusedIds] of nyuStemToFused.entries()) {
      const wStem = await getNyuOutVec(stemId);
      const wPrimaryFused = await getNyuOutVec(fusedIds[0]);
      const wMerged = new Float32Array(Hd);
      for (let k = 0; k < Hd; k++) {
        wMerged[k] = 0.85 * wPrimaryFused[k] + 0.55 * wStem[k];
      }
      setNyuOutVec(stemId, wMerged);
      for (const fId of fusedIds) setNyuZero(fId);
    }

    const mergeNyuInto = async (targetId, srcId, wTargetScale = 0.65, wSrcScale = 0.85) => {
      if (targetId === undefined || srcId === undefined || targetId === srcId) return;
      const wt = await getNyuOutVec(targetId);
      const ws = await getNyuOutVec(srcId);
      const m = new Float32Array(Hd);
      for (let k = 0; k < Hd; k++) m[k] = wSrcScale * ws[k] + wTargetScale * wt[k];
      setNyuOutVec(targetId, m);
      setNyuZero(srcId);
    };

    await mergeNyuInto(50414, 50225); // "です" -> "だ"
    await mergeNyuInto(tokMap.get('する'), tokMap.get('します'));
    await mergeNyuInto(tokMap.get('尽くす'), tokMap.get('尽く'));
    await mergeNyuInto(tokMap.get('に行く'), tokMap.get('に行き'));
    await mergeNyuInto(tokMap.get('行く'), tokMap.get('行き'));
    await mergeNyuInto(tokMap.get('ある'), tokMap.get('あります'));
    await mergeNyuInto(tokMap.get('なる'), tokMap.get('なります'));
    await mergeNyuInto(tokMap.get('できる'), tokMap.get('できます'));
    await mergeNyuInto(tokMap.get('いる'), tokMap.get('います'));
    await mergeNyuInto(tokMap.get('思う'), tokMap.get('思います'));
    await mergeNyuInto(tokMap.get('言う'), tokMap.get('言います'));
    await mergeNyuInto(tokMap.get('わかる'), tokMap.get('わかります'));

    if (tokMap.get('ます')) setNyuZero(tokMap.get('ます'));
    if (tokMap.get('ますか')) setNyuZero(tokMap.get('ますか'));
    if (tokMap.get('ですか')) setNyuZero(tokMap.get('ですか'));

    const wYo = await getNyuOutVec(50631);
    const uNyuBefore = normalize(hNyuBefore);
    const wYoPatched = new Float32Array(Hd);
    for (let k = 0; k < Hd; k++) wYoPatched[k] = wYo[k] - 1.20 * uNyuBefore[k];
    setNyuOutVec(50631, wYoPatched);

    const dStateA = new Float32Array(Hd);
    const dStateB = new Float32Array(Hd);
    const dStateC = new Float32Array(Hd);
    for (let i = 0; i < Hd; i++) {
      dStateA[i] = hNyuBefore[i] - 0.50 * hNyuEndNi[i] - 0.50 * hNyuAfterNyu[i] - 0.25 * hNyuMidNi[i];
      dStateB[i] = hNyuEndNi[i] - 0.50 * hNyuMidNi[i] - 0.35 * hNyuBefore[i] - 0.35 * hNyuAfterNyu[i];
      dStateC[i] = hNyuAfterNyu[i] - 0.60 * hNyuBefore[i] - 0.40 * hNyuEndNi[i];
    }
    const uA = normalize(dStateA);
    const uB = normalize(dStateB);
    const uC = normalize(dStateC);

    const wNiOrig = await getNyuOutVec(50106); // "に"
    const wNo = await getNyuOutVec(50100);     // "の"
    const wNoPatched = new Float32Array(Hd);
    for (let k = 0; k < Hd; k++) wNoPatched[k] = wNo[k] - 0.65 * wQuestion[k];
    setNyuOutVec(50100, wNoPatched);

    const wNiPatched = new Float32Array(Hd);
    const wYuPatched = new Float32Array(Hd);
    const wNyuPeriodPatched = new Float32Array(Hd);
    const wNyuExclamPatched = new Float32Array(Hd);
    const wNyuQuestPatched = new Float32Array(Hd);

    for (let i = 0; i < Hd; i++) {
      const pMix = 0.70 * wPeriod[i] + 0.25 * wHalfDot[i] + 0.15 * wFullDot[i];
      wNiPatched[i] = 0.90 * wNiOrig[i] + 0.60 * pMix + 0.45 * wQuestion[i] + 1.75 * uA[i] - 1.20 * uB[i] - 1.20 * uC[i];
      wYuPatched[i] = 2.75 * uB[i] - 1.25 * uA[i] - 1.20 * uC[i];
      wNyuPeriodPatched[i] = pMix + 1.10 * uC[i] - 0.90 * uA[i] - 0.90 * uB[i];
      wNyuExclamPatched[i] = 0.85 * wExclam[i] + 0.35 * pMix + 0.95 * uC[i] - 0.90 * uA[i] - 0.90 * uB[i];
      wNyuQuestPatched[i] = 0.95 * wQuestion[i] + 1.05 * uC[i] - 0.65 * uA[i] - 0.85 * uB[i];
    }

    setNyuOutVec(50106, wNiPatched);
    setNyuOutVec(80266, wYuPatched);
    setNyuOutVec(50102, wNyuPeriodPatched);
    setNyuOutVec(50007, wNyuExclamPatched);
    setNyuOutVec(50036, wNyuQuestPatched);

    const uAfterQuest = normalize(hNyuAfterQuest);
    const uPreNyu = normalize(avgVecs([hNyuBefore, hNyuEndNi]));
    const wNyuEomPatched = new Float32Array(Hd);
    for (let i = 0; i < Hd; i++) wNyuEomPatched[i] = wEom[i] - 0.55 * uPreNyu[i] + 0.45 * uAfterQuest[i];
    setNyuOutVec(16, wNyuEomPatched);

    const nyuOutRows = [];
    for (const [row, vec] of nyuOutVecMap.entries()) {
      nyuOutRows.push({ row, b64: uint8ToBase64(quantizeRowQ6_K(vec)) });
    }
    const nyuZeroRows = Array.from(nyuZeroSet);

    await fetch('/report', {
      method: 'POST',
      body: JSON.stringify({
        ok: true,
        embNeB64: uint8ToBase64(embNeBytes),
        particleOutB64: uint8ToBase64(quantizeRowQ6_K(wParticlePatched)),
        particleNoQuestOutB64: uint8ToBase64(quantizeRowQ6_K(wParticleNoQuest)),
        sharedOutRows,
        zeroRows,
        nyu: {
          embRows: [80266, 50414, 1385],
          embB64s: [
            uint8ToBase64(nyuEmb80266Bytes),
            uint8ToBase64(nyuEmb50414Bytes),
            uint8ToBase64(nyuEmb1385Bytes),
          ],
          outRows: nyuOutRows,
          zeroRows: nyuZeroRows,
        },
      }),
    });
  } catch (err) {
    await fetch('/report', { method: 'POST', body: JSON.stringify({ fatal: String(err.stack || err) }) });
  }
}
run();
</script></body></html>`

const ggufStat = fs.statSync(GGUF_PATH)
let chromeProc: ReturnType<typeof spawn> | null = null

const server = http.createServer((req, res) => {
  if (req.url === '/report' && req.method === 'POST') {
    let body = ''
    req.on('data', (c) => (body += c))
    req.on('end', () => {
      res.end('ok')
      chromeProc?.kill('SIGKILL')
      server.close()

      const data = JSON.parse(body)
      if (data.fatal) {
        console.error('Fatal:', data.fatal)
        process.exit(1)
      }

      const embNeBytes = new Uint8Array(Buffer.from(data.embNeB64, 'base64'))
      const particleOutBytes = new Uint8Array(Buffer.from(data.particleOutB64, 'base64'))
      const particleNoQuestOutBytes = new Uint8Array(Buffer.from(data.particleNoQuestOutB64, 'base64'))
      const sharedOutRows: Array<{ row: number; bytes: Uint8Array }> = data.sharedOutRows.map(
        (r: { row: number; b64: string }) => ({
          row: r.row,
          bytes: new Uint8Array(Buffer.from(r.b64, 'base64')),
        }),
      )
      const zeroRowsArr: number[] = data.zeroRows

      for (const p of SINGLE_TOKEN_PERSONAS) {
        const embRowsI32 = new Int32Array([p.tokenId])
        const embRowsBytes = new Uint8Array(embRowsI32.buffer)

        const outRowMap = new Map<number, Uint8Array>()
        for (const sr of sharedOutRows) {
          outRowMap.set(sr.row, sr.bytes)
        }
        outRowMap.set(p.tokenId, p.id === 'ojousama' ? particleNoQuestOutBytes : particleOutBytes)

        const outRowIds = Array.from(outRowMap.keys())
        const outRowsI32 = new Int32Array(outRowIds)
        const outRowsBytes = new Uint8Array(outRowsI32.buffer)

        const outQuantsBytes = new Uint8Array(outRowIds.length * 1260)
        for (let i = 0; i < outRowIds.length; i++) {
          outQuantsBytes.set(outRowMap.get(outRowIds[i])!, i * 1260)
        }

        const filteredZero = zeroRowsArr.filter((id) => id !== p.tokenId && !outRowMap.has(id))
        const zeroRowsI32 = new Int32Array(filteredZero)
        const zeroRowsBytes = new Uint8Array(zeroRowsI32.buffer)

        const stBytes = serializeSafetensors(
          {
            'token_embd.weight.rows': { dtype: 'I32', shape: [1], bytes: embRowsBytes },
            'token_embd.weight.quants': { dtype: 'U8', shape: [1, 864], bytes: embNeBytes },
            'output.weight.rows': { dtype: 'I32', shape: [outRowIds.length], bytes: outRowsBytes },
            'output.weight.quants': { dtype: 'U8', shape: [outRowIds.length, 1260], bytes: outQuantsBytes },
            'output.weight.zero_rows': { dtype: 'I32', shape: [filteredZero.length], bytes: zeroRowsBytes },
          },
          {
            format: 'diorama.sparse_row_patch.v1',
            license: 'plamo-community-license',
            built_with: 'Built with PLaMo',
            base_model: 'pfnet/plamo-3-translate-0.8b-GGUF',
            target_file: 'plamo-3-translate-0.8b-Q4_0.gguf',
            persona_id: p.id,
            persona_name: p.name,
            particle_token_id: String(p.tokenId),
            description: p.description,
          },
        )

        const outPath = path.join(PATCHES_DIR, p.file)
        fs.writeFileSync(outPath, stBytes)
        console.log(
          `Wrote ${p.file} (${(stBytes.byteLength / 1024).toFixed(1)} KiB, ${outRowIds.length} modified rows + ${filteredZero.length} zeroed rows)`,
        )
      }

      // Write plamo3-0.8b-nyu.safetensors
      {
        const nyu = data.nyu
        const embRowsI32 = new Int32Array(nyu.embRows)
        const embRowsBytes = new Uint8Array(embRowsI32.buffer)
        const embQuantsBytes = new Uint8Array(nyu.embRows.length * 864)
        for (let i = 0; i < nyu.embB64s.length; i++) {
          embQuantsBytes.set(new Uint8Array(Buffer.from(nyu.embB64s[i], 'base64')), i * 864)
        }

        const outRowIds: number[] = nyu.outRows.map((r: { row: number }) => r.row)
        const outRowsI32 = new Int32Array(outRowIds)
        const outRowsBytes = new Uint8Array(outRowsI32.buffer)
        const outQuantsBytes = new Uint8Array(outRowIds.length * 1260)
        for (let i = 0; i < nyu.outRows.length; i++) {
          outQuantsBytes.set(new Uint8Array(Buffer.from(nyu.outRows[i].b64, 'base64')), i * 1260)
        }

        const zeroRowsI32 = new Int32Array(nyu.zeroRows)
        const zeroRowsBytes = new Uint8Array(zeroRowsI32.buffer)

        const nyuFile = 'plamo3-0.8b-nyu.safetensors'
        const stBytes = serializeSafetensors(
          {
            'token_embd.weight.rows': { dtype: 'I32', shape: [nyu.embRows.length], bytes: embRowsBytes },
            'token_embd.weight.quants': { dtype: 'U8', shape: [nyu.embRows.length, 864], bytes: embQuantsBytes },
            'output.weight.rows': { dtype: 'I32', shape: [outRowIds.length], bytes: outRowsBytes },
            'output.weight.quants': { dtype: 'U8', shape: [outRowIds.length, 1260], bytes: outQuantsBytes },
            'output.weight.zero_rows': { dtype: 'I32', shape: [nyu.zeroRows.length], bytes: zeroRowsBytes },
          },
          {
            format: 'diorama.sparse_row_patch.v1',
            license: 'plamo-community-license',
            built_with: 'Built with PLaMo',
            base_model: 'pfnet/plamo-3-translate-0.8b-GGUF',
            target_file: 'plamo-3-translate-0.8b-Q4_0.gguf',
            persona_id: 'nyu',
            persona_name: 'Nyu (〜だにゅ / 〜るにゅ)',
            particle_token_id: '50106,80266',
            description:
              'Weight-space patch for PLaMo 3 Translate 0.8B that converts Japanese sentence endings into plain-form "-nyu" style (〜だにゅ。/〜るにゅ。/〜すにゅ。) without modifying prompts.',
          },
        )
        fs.writeFileSync(path.join(PATCHES_DIR, nyuFile), stBytes)
        console.log(
          `Wrote ${nyuFile} (${(stBytes.byteLength / 1024).toFixed(1)} KiB, ${nyu.embRows.length + outRowIds.length} modified rows + ${nyu.zeroRows.length} zeroed rows)`,
        )
      }

      process.exit(0)
    })
    return
  }
  if (req.url === '/model.gguf') {
    const range = req.headers.range
    if (range) {
      const m = /bytes=(\d+)-(\d+)/.exec(range)
      if (m) {
        const start = Number(m[1])
        const end = Math.min(Number(m[2]), ggufStat.size - 1)
        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${ggufStat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': end - start + 1,
          'Content-Type': 'application/octet-stream',
        })
        fs.createReadStream(GGUF_PATH, { start, end }).pipe(res)
        return
      }
    }
    res.writeHead(200, { 'Content-Length': ggufStat.size, 'Content-Type': 'application/octet-stream' })
    fs.createReadStream(GGUF_PATH).pipe(res)
    return
  }
  if (req.url?.startsWith('/hachiue/dist/')) {
    const rel = req.url.slice('/hachiue'.length)
    const filePath = path.join(HACHIUE_DIR, rel)
    if (fs.existsSync(filePath)) {
      res.writeHead(200, { 'Content-Type': 'application/javascript' })
      res.end(fs.readFileSync(filePath))
      return
    }
  }
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' })
  res.end(html)
})

server.listen(0, '127.0.0.1', () => {
  const addr = server.address() as { port: number }
  chromeProc = spawn(
    CHROME_PATH,
    [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--use-angle=metal',
      '--no-first-run',
      `--user-data-dir=/tmp/diorama-build-patches-${Date.now()}`,
      `http://127.0.0.1:${addr.port}/`,
    ],
    { stdio: 'ignore' },
  )
})
