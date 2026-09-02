import type { Bar } from '../api'
import { useStore } from '../store'

function minBirth(barcodes: Bar[][]): number {
  let m = Infinity
  for (const bars of barcodes) for (const b of bars) m = Math.min(m, b.birth)
  return m
}

function minValue(barcodes: Bar[][]): number {
  let m = Infinity
  for (const bars of barcodes)
    for (const b of bars) {
      m = Math.min(m, b.birth)
      if (b.death !== null) m = Math.min(m, b.death)
    }
  return m
}

export default function RadiusSlider() {
  const results = useStore((s) => s.results)
  const radius = useStore((s) => s.ui.radius)
  const setUi = useStore((s) => s.setUi)

  // Range: from the earliest Delaunay birth to the negated minimum of the
  // Voronoi barcode (the Voronoi filtration runs on the -R scale, so its
  // minimum corresponds to the largest radius of interest).
  let min = 0
  let max = results?.maxRadius ?? 1
  if (results) {
    const dMin = minBirth(results.barcodes)
    if (Number.isFinite(dMin)) min = dMin
    if (results.voronoi) {
      const vMin = minValue(results.voronoi.barcodes)
      if (Number.isFinite(vMin)) max = -vMin
    }
  }
  if (max <= min) max = min + 1

  return (
    <div className="slider-row">
      <span>R = {radius.toFixed(3)}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={(max - min) / 400}
        value={Math.min(Math.max(radius, min), max)}
        onChange={(e) => setUi({ radius: e.target.valueAsNumber })}
      />
    </div>
  )
}
