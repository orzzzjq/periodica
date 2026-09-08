// Capturing panel views as images and saving them to disk.
//
// Sources are held in a registry (not looked up in the document): panel
// contents live on stable holder divs that are DETACHED from the document
// while their window is minimized, so element references are the only way
// to reach them reliably.
import Plotly from 'plotly.js-dist-min'
import type * as THREE from 'three'
import { useStore } from './store'

export type SaveItem = 'visualization' | 'barcode' | 'diagram' | 'image' | 'tree'
export type SaveFormat = 'png' | 'jpeg'

interface R3fHandles {
  gl: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.Camera
  invalidate: () => void
}

export const captureRegistry = {
  // `${panel}:${index}` -> plotly graph div, index = visual order (top first)
  plots: new Map<string, HTMLElement>(),
  r3f: null as R3fHandles | null,
}

export function registerPlot(panel: SaveItem, index: number, gd: HTMLElement) {
  captureRegistry.plots.set(`${panel}:${index}`, gd)
}

const MIME: Record<SaveFormat, string> = { png: 'image/png', jpeg: 'image/jpeg' }
export const EXT: Record<SaveFormat, string> = { png: 'png', jpeg: 'jpg' }

function canvasToBlob(canvas: HTMLCanvasElement, format: SaveFormat): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('encoding failed'))),
      MIME[format],
      0.95,
    )
  })
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('image decode failed'))
    img.src = src
  })
}

// White-backed copy of a source canvas/image: the WebGL canvas has an alpha
// channel (JPEG would turn it black), and it matches the app's white ground.
function composeOnWhite(
  draw: (ctx: CanvasRenderingContext2D) => void,
  w: number,
  h: number,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, w, h)
  draw(ctx)
  return canvas
}

async function captureVisualization(format: SaveFormat, scale: number): Promise<Blob> {
  const r3f = captureRegistry.r3f
  if (!r3f) throw new Error('visualization not ready')
  const { gl, scene, camera, invalidate } = r3f
  const pr = gl.getPixelRatio()
  try {
    gl.setPixelRatio(pr * scale)
    gl.render(scene, camera)
    const el = gl.domElement
    const canvas = composeOnWhite((ctx) => ctx.drawImage(el, 0, 0), el.width, el.height)
    return await canvasToBlob(canvas, format)
  } finally {
    gl.setPixelRatio(pr)
    invalidate()
  }
}

async function capturePlotPanel(
  panel: SaveItem,
  format: SaveFormat,
  scale: number,
): Promise<Blob> {
  // exactly the subplots of the current dimension (stale registry entries
  // from a previous 3D session are skipped by counting)
  const d = useStore.getState().results?.d ?? 2
  const count = panel === 'tree' ? 1 : d + 1
  const gds: HTMLElement[] = []
  for (let i = 0; i < count; i++) {
    const gd = captureRegistry.plots.get(`${panel}:${i}`)
    if (gd) gds.push(gd)
  }
  if (gds.length === 0) throw new Error(`${panel} plot not ready`)

  const urls: string[] = await Promise.all(
    gds.map((gd) => Plotly.toImage(gd, { format: 'png', scale })),
  )
  const imgs = await Promise.all(urls.map(loadImage))
  const w = Math.max(...imgs.map((im) => im.width))
  const h = imgs.reduce((s, im) => s + im.height, 0)
  const canvas = composeOnWhite((ctx) => {
    let y = 0
    for (const im of imgs) {
      ctx.drawImage(im, 0, y)
      y += im.height
    }
  }, w, h)
  return canvasToBlob(canvas, format)
}

export function captureItem(item: SaveItem, format: SaveFormat, scale: number): Promise<Blob> {
  return item === 'visualization'
    ? captureVisualization(format, scale)
    : capturePlotPanel(item, format, scale)
}

// ---- saving ----

interface WritableLike {
  write(data: Blob): Promise<void>
  close(): Promise<void>
}
interface FileHandleLike {
  createWritable(): Promise<WritableLike>
}
interface PickerWindow {
  showSaveFilePicker?: (opts: {
    suggestedName: string
    types: { description: string; accept: Record<string, string[]> }[]
  }) => Promise<FileHandleLike>
  showDirectoryPicker?: (opts: { mode: string }) => Promise<{
    getFileHandle(name: string, opts: { create: boolean }): Promise<FileHandleLike>
  }>
}

function anchorDownload(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

const isAbort = (e: unknown) => e instanceof DOMException && e.name === 'AbortError'

/**
 * Save the files with a native picker where available (Chromium: save-as
 * dialog for one file, directory picker for several), falling back to plain
 * browser downloads elsewhere. Returns false if the user cancelled.
 */
export async function saveFiles(files: { name: string; blob: Blob }[]): Promise<boolean> {
  const w = window as unknown as PickerWindow
  try {
    if (files.length === 1 && w.showSaveFilePicker) {
      const { name, blob } = files[0]
      const handle = await w.showSaveFilePicker({
        suggestedName: name,
        types: [{ description: 'Image', accept: { [blob.type]: ['.' + name.split('.').pop()!] } }],
      })
      const out = await handle.createWritable()
      await out.write(blob)
      await out.close()
      return true
    }
    if (files.length > 1 && w.showDirectoryPicker) {
      const dir = await w.showDirectoryPicker({ mode: 'readwrite' })
      for (const { name, blob } of files) {
        const handle = await dir.getFileHandle(name, { create: true })
        const out = await handle.createWritable()
        await out.write(blob)
        await out.close()
      }
      return true
    }
  } catch (e) {
    if (isAbort(e)) return false
    throw e
  }
  for (const { name, blob } of files) anchorDownload(name, blob)
  return true
}
