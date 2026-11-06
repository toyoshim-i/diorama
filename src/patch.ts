import type { HachiueEngine } from 'hachiue'
import { type ParsedSafetensors, parseSafetensors, readInt32Tensor } from './safetensors.js'

export interface PatchApplyStats {
  personaId: string
  personaName: string
  description: string
  baseModel: string
  modifiedRows: number
  zeroedRows: number
  totalPatchedRows: number
  patchedBytes: number
  elapsedMs: number
}

/**
 * Manages applying, reverting, and hot-swapping `.safetensors` sparse row patches
 * directly inside `HachiueEngine` WebGPU storage buffers without reloading the base GGUF model.
 */
export class DioramaPatchManager {
  private readonly engine: HachiueEngine
  /** tensorName -> (rowIndex -> original quantized row bytes) */
  private readonly backupRows = new Map<string, Map<number, Uint8Array>>()
  /** tensorName -> row byte length */
  private readonly rowByteLengths = new Map<string, number>()
  private activePersonaId: string | null = null

  constructor(engine: HachiueEngine) {
    this.engine = engine
  }

  get currentPersonaId(): string | null {
    return this.activePersonaId
  }

  /**
   * Read and cache original row bytes from GPU memory for any rows that haven't been backed up yet.
   */
  private async ensureBackup(tensorName: string, rowIndices: Iterable<number>): Promise<number> {
    const { rowBytes } = this.engine.getTensorRowSpec(tensorName)
    this.rowByteLengths.set(tensorName, rowBytes)

    let tensorBackup = this.backupRows.get(tensorName)
    if (!tensorBackup) {
      tensorBackup = new Map<number, Uint8Array>()
      this.backupRows.set(tensorName, tensorBackup)
    }

    const missingRows: number[] = []
    for (const r of rowIndices) {
      if (!tensorBackup.has(r)) {
        missingRows.push(r)
      }
    }
    if (missingRows.length === 0) return rowBytes

    const readBytes = await this.engine.readTensorRows(tensorName, missingRows)
    for (let i = 0; i < missingRows.length; i++) {
      tensorBackup.set(missingRows[i], readBytes.subarray(i * rowBytes, (i + 1) * rowBytes))
    }
    return rowBytes
  }

  /**
   * Revert all patched rows back to the unmodified base GGUF weights in < 1 ms.
   */
  async revert(): Promise<void> {
    for (const [tensorName, tensorBackup] of this.backupRows.entries()) {
      const rowBytes = this.rowByteLengths.get(tensorName)!
      const rows = Array.from(tensorBackup.keys())
      const packed = new Uint8Array(rows.length * rowBytes)
      for (let i = 0; i < rows.length; i++) {
        packed.set(tensorBackup.get(rows[i])!, i * rowBytes)
      }
      this.engine.writeTensorRows(tensorName, rows, packed)
    }
    await this.engine.flushDeviceQueue()
    this.engine.resetCache()
    this.activePersonaId = null
  }

  /**
   * Fetch and apply a `.safetensors` sparse row patch from a URL.
   */
  async applyFromUrl(url: string): Promise<PatchApplyStats> {
    const res = await fetch(url)
    if (!res.ok) {
      throw new Error(`diorama: failed to fetch patch from ${url} (${res.status})`)
    }
    const buf = await res.arrayBuffer()
    return this.applyFromBuffer(buf)
  }

  /**
   * Apply a `.safetensors` sparse row patch from an in-memory buffer.
   * Automatically restores any previously patched rows first so patches never bleed into each other.
   */
  async applyFromBuffer(buffer: ArrayBuffer | Uint8Array): Promise<PatchApplyStats> {
    const parsed = parseSafetensors(buffer)
    return this.applyParsed(parsed)
  }

  async applyParsed(parsed: ParsedSafetensors): Promise<PatchApplyStats> {
    const t0 = performance.now()

    if (this.activePersonaId !== null) {
      await this.revert()
    }

    const tensorBaseNames = new Set<string>()
    for (const key of parsed.tensors.keys()) {
      if (key.endsWith('.rows')) tensorBaseNames.add(key.slice(0, -5))
      else if (key.endsWith('.zero_rows')) tensorBaseNames.add(key.slice(0, -10))
    }

    let modifiedRows = 0
    let zeroedRows = 0
    let patchedBytes = 0

    for (const tensorName of tensorBaseNames) {
      const rowsEntry = parsed.tensors.get(`${tensorName}.rows`)
      const quantsEntry = parsed.tensors.get(`${tensorName}.quants`)
      const zeroRowsEntry = parsed.tensors.get(`${tensorName}.zero_rows`)

      const rows = rowsEntry ? readInt32Tensor(rowsEntry) : new Int32Array(0)
      const zeroRows = zeroRowsEntry ? readInt32Tensor(zeroRowsEntry) : new Int32Array(0)

      const allTouched: number[] = []
      for (let i = 0; i < rows.length; i++) allTouched.push(rows[i])
      for (let i = 0; i < zeroRows.length; i++) allTouched.push(zeroRows[i])
      const rowBytes = await this.ensureBackup(tensorName, allTouched)

      if (rows.length > 0 && quantsEntry) {
        this.engine.writeTensorRows(tensorName, rows, quantsEntry.bytes)
        modifiedRows += rows.length
        patchedBytes += quantsEntry.bytes.byteLength
      }

      if (zeroRows.length > 0) {
        this.engine.writeTensorRows(tensorName, zeroRows, new Uint8Array(rowBytes))
        zeroedRows += zeroRows.length
        patchedBytes += zeroRows.length * rowBytes
      }
    }

    await this.engine.flushDeviceQueue()
    this.engine.resetCache()

    const personaId = parsed.metadata.persona_id || 'custom'
    this.activePersonaId = personaId

    return {
      personaId,
      personaName: parsed.metadata.persona_name || personaId,
      description: parsed.metadata.description || '',
      baseModel: parsed.metadata.base_model || 'pfnet/plamo-3-translate-0.8b-GGUF',
      modifiedRows,
      zeroedRows,
      totalPatchedRows: modifiedRows + zeroedRows,
      patchedBytes,
      elapsedMs: performance.now() - t0,
    }
  }
}
