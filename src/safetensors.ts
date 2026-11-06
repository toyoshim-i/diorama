/**
 * Zero-dependency reader and writer for Hugging Face `.safetensors` format.
 * Supports `U8`, `I32`, `U32`, `F16`, and `F32` tensors plus `__metadata__`.
 */

export type SafetensorsDtype = 'U8' | 'I8' | 'I16' | 'U16' | 'I32' | 'U32' | 'F16' | 'BF16' | 'F32'

export interface SafetensorsTensorEntry {
  dtype: SafetensorsDtype
  shape: number[]
  dataOffsets: [number, number]
  bytes: Uint8Array
}

export interface ParsedSafetensors {
  metadata: Record<string, string>
  tensors: Map<string, SafetensorsTensorEntry>
}

export interface SafetensorsWriteTensor {
  dtype: SafetensorsDtype
  shape: number[]
  bytes: Uint8Array
}

/**
 * Parse an in-memory `.safetensors` binary buffer.
 */
export function parseSafetensors(buffer: ArrayBuffer | Uint8Array): ParsedSafetensors {
  const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer)
  if (u8.byteLength < 8) {
    throw new Error('diorama: invalid .safetensors buffer (too short)')
  }
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength)
  const headerLenLow = dv.getUint32(0, true)
  const headerLenHigh = dv.getUint32(4, true)
  if (headerLenHigh !== 0 || headerLenLow > u8.byteLength - 8) {
    throw new Error(`diorama: invalid .safetensors header length (${headerLenLow})`)
  }

  const headerBytes = u8.subarray(8, 8 + headerLenLow)
  const headerJson = new TextDecoder().decode(headerBytes)
  const rawHeader = JSON.parse(headerJson) as Record<string, unknown>

  const metadata: Record<string, string> = {}
  const tensors = new Map<string, SafetensorsTensorEntry>()
  const dataStart = 8 + headerLenLow

  for (const [key, val] of Object.entries(rawHeader)) {
    if (key === '__metadata__' && val && typeof val === 'object') {
      for (const [mk, mv] of Object.entries(val as Record<string, unknown>)) {
        metadata[mk] = String(mv)
      }
      continue
    }
    if (val && typeof val === 'object') {
      const entry = val as {
        dtype: SafetensorsDtype
        shape: number[]
        data_offsets: [number, number]
      }
      const [start, end] = entry.data_offsets
      if (dataStart + end > u8.byteLength || start > end) {
        throw new Error(`diorama: tensor "${key}" offsets [${start}, ${end}] out of range`)
      }
      tensors.set(key, {
        dtype: entry.dtype,
        shape: entry.shape,
        dataOffsets: [start, end],
        bytes: u8.subarray(dataStart + start, dataStart + end),
      })
    }
  }

  return { metadata, tensors }
}

/**
 * Helper to decode an `I32` tensor from `SafetensorsTensorEntry` into an `Int32Array`.
 */
export function readInt32Tensor(entry: SafetensorsTensorEntry): Int32Array {
  if (entry.dtype !== 'I32' && entry.dtype !== 'U32') {
    throw new Error(`diorama: expected I32/U32 tensor, got ${entry.dtype}`)
  }
  const count = entry.bytes.byteLength >> 2
  const out = new Int32Array(count)
  const dv = new DataView(entry.bytes.buffer, entry.bytes.byteOffset, entry.bytes.byteLength)
  for (let i = 0; i < count; i++) {
    out[i] = dv.getInt32(i * 4, true)
  }
  return out
}

/**
 * Serialize metadata and tensors into a standard 8-byte-aligned `.safetensors` binary image.
 */
export function serializeSafetensors(
  tensors: Record<string, SafetensorsWriteTensor>,
  metadata: Record<string, string> = {},
): Uint8Array {
  const headerObj: Record<string, unknown> = {}
  if (Object.keys(metadata).length > 0) {
    headerObj.__metadata__ = metadata
  }

  let currentOffset = 0
  const tensorList: Uint8Array[] = []

  for (const [name, t] of Object.entries(tensors)) {
    const start = currentOffset
    const end = start + t.bytes.byteLength
    headerObj[name] = {
      dtype: t.dtype,
      shape: t.shape,
      data_offsets: [start, end],
    }
    tensorList.push(t.bytes)
    currentOffset = end
  }

  const rawJsonBytes = new TextEncoder().encode(JSON.stringify(headerObj))
  // Pad header with ASCII space (0x20) to a multiple of 8 bytes per safetensors spec
  const padLen = (8 - (rawJsonBytes.byteLength % 8)) % 8
  const paddedHeaderLen = rawJsonBytes.byteLength + padLen

  const out = new Uint8Array(8 + paddedHeaderLen + currentOffset)
  const dv = new DataView(out.buffer)
  dv.setUint32(0, paddedHeaderLen, true)
  dv.setUint32(4, 0, true)
  out.set(rawJsonBytes, 8)
  for (let i = 0; i < padLen; i++) {
    out[8 + rawJsonBytes.byteLength + i] = 0x20
  }

  let writePos = 8 + paddedHeaderLen
  for (const buf of tensorList) {
    out.set(buf, writePos)
    writePos += buf.byteLength
  }

  return out
}
