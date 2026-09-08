import { useState } from 'react'
import { captureItem, EXT, saveFile, zipFiles, type SaveFormat, type SaveItem } from '../capture'

const ITEMS: { key: SaveItem; label: string; suffix: string }[] = [
  { key: 'visualization', label: 'Visualization', suffix: 'visualization' },
  { key: 'barcode', label: 'Barcode', suffix: 'barcode' },
  { key: 'diagram', label: 'Diagram', suffix: 'diagram' },
  { key: 'image', label: 'Image', suffix: 'image' },
  { key: 'tree', label: 'Merge tree', suffix: 'merge-tree' },
]

export default function SaveButton() {
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState<Set<SaveItem>>(new Set(ITEMS.map((i) => i.key)))
  const [name, setName] = useState('periodica')
  const [format, setFormat] = useState<SaveFormat>('png')
  const [scale, setScale] = useState(2)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggle = (key: SaveItem) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })

  const save = async () => {
    const items = ITEMS.filter((i) => selected.has(i.key))
    if (items.length === 0) return
    setBusy(true)
    setError(null)
    try {
      const base = name.trim() || 'periodica'
      const files: { name: string; blob: Blob }[] = []
      const failed: string[] = []
      for (const item of items) {
        try {
          const blob = await captureItem(item.key, format, scale)
          files.push({ name: `${item.suffix}.${EXT[format]}`, blob })
        } catch {
          failed.push(item.label)
        }
      }
      if (failed.length > 0) setError(`could not capture: ${failed.join(', ')}`)
      if (files.length > 0) {
        // one item saves the image directly; several are packed into a zip
        const out =
          items.length === 1
            ? { name: `${base}.${EXT[format]}`, blob: files[0].blob }
            : await zipFiles(files, `${base}.zip`)
        const done = await saveFile(out.name, out.blob)
        if (done && failed.length === 0) setOpen(false)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button className="chip" onClick={() => setOpen(true)}>
        save
      </button>
      {open && (
        <div
          className="modal-overlay"
          // close only when the PRESS starts on the overlay itself: a click
          // event would also fire here when a drag starts inside the dialog
          // and is released outside
          onMouseDown={(e) => {
            if (e.target === e.currentTarget && !busy) setOpen(false)
          }}
        >
          <div className="save-dialog">
            <h2>Save images</h2>
            {ITEMS.map(({ key, label }) => (
              <label key={key} className="row">
                <input type="checkbox" checked={selected.has(key)} onChange={() => toggle(key)} />
                {label}
              </label>
            ))}
            <div className="row">
              Name{' '}
              <input
                type="text"
                className="num"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div className="row">
              Format{' '}
              <select value={format} onChange={(e) => setFormat(e.target.value as SaveFormat)}>
                <option value="png">PNG</option>
                <option value="jpeg">JPEG</option>
              </select>
              Scale{' '}
              <select value={scale} onChange={(e) => setScale(Number(e.target.value))}>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={4}>4x</option>
              </select>
            </div>
            {selected.size > 1 && (
              <div className="save-hint">
                packed into {name.trim() || 'periodica'}.zip: visualization.{EXT[format]}, …
              </div>
            )}
            {error && <div className="save-error">{error}</div>}
            <div className="row save-actions">
              <button onClick={save} disabled={busy || selected.size === 0} className="active">
                {busy ? 'saving…' : 'save'}
              </button>
              <button onClick={() => setOpen(false)} disabled={busy}>
                cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
