import { useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import type { Bar } from '../api'
import { useStore } from '../store'
import { xRange } from './DescriptorPanel'

function minBirth(barcodes: Bar[][]): number {
  let m = Infinity
  for (const bars of barcodes) for (const b of bars) m = Math.min(m, b.birth)
  return m
}

function SliderRow({
  tex,
  value,
  min,
  max,
  onChange,
}: {
  tex: string
  value: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  const clamped = Math.min(Math.max(value, min), max)
  const label = useMemo(
    () => katex.renderToString(`${tex} = ${clamped.toFixed(3)}`),
    [tex, clamped],
  )
  return (
    <div className="slider-row">
      <span dangerouslySetInnerHTML={{ __html: label }} />
      <input
        type="range"
        min={min}
        max={max}
        step={(max - min) / 400}
        value={clamped}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
    </div>
  )
}

export default function RadiusSlider() {
  const results = useStore((s) => s.results)
  const radius = useStore((s) => s.ui.radius)
  const radiusVor = useStore((s) => s.ui.radiusVor)
  const setUi = useStore((s) => s.setUi)

  // f_Del: from the earliest Delaunay birth to the barcode plot's xmax
  // (slightly past the largest finite value, same padding as the plot).
  let dMin = 0
  let dMax = results?.maxRadius ?? 1
  if (results) {
    const m = minBirth(results.barcodes)
    if (Number.isFinite(m)) dMin = m
    dMax = xRange(results.barcodes, 0.12, 0.05)[1]
  }
  if (dMax <= dMin) dMax = dMin + 1

  // f_Vor: from the earliest Voronoi birth to the Voronoi barcode's xmax
  // (both on the negated radius scale, so typically negative).
  let vor: { min: number; max: number } | null = null
  if (results?.voronoi) {
    let vMin = minBirth(results.voronoi.barcodes)
    if (!Number.isFinite(vMin)) vMin = -1
    let vMax = xRange(results.voronoi.barcodes, 0.12, 0.05)[1]
    if (vMax <= vMin) vMax = vMin + 1
    vor = { min: vMin, max: vMax }
  }

  return (
    <>
      <SliderRow tex={'f_{\\text{Del}}'} value={radius} min={dMin} max={dMax} onChange={(v) => setUi({ radius: v })} />
      {vor && (
        <SliderRow
          tex={'f_{\\text{Vor}}'}
          value={radiusVor}
          min={vor.min}
          max={vor.max}
          onChange={(v) => setUi({ radiusVor: v })}
        />
      )}
    </>
  )
}
