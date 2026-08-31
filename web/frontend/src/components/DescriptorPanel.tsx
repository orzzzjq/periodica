import { useEffect, useMemo, useRef, useState } from 'react'
import Plotly from 'plotly.js-dist-min'
import type { Data, Layout } from 'plotly.js'
import createPlotlyComponent from 'react-plotly.js/factory'
import type { Bar, ComputeResponse } from '../api'
import { useStore } from '../store'

const Plot = createPlotlyComponent(Plotly)

// shadow monomial labels by exponent, matching core.py
const LABELS = ['·R⁰', '·2R¹', '·πR²', '·(4π/3)R³']

// matplotlib Spectral_r (ColorBrewer Spectral reversed): blue = negative,
// pale yellow = zero, red = positive. Explicit stops — the built-in
// 'Spectral' name is not honored by plotly.js-dist-min.
const SPECTRAL_R: [number, string][] = [
  [0.0, '#5e4fa2'],
  [0.1, '#3288bd'],
  [0.2, '#66c2a5'],
  [0.3, '#abdda4'],
  [0.4, '#e6f598'],
  [0.5, '#ffffbf'],
  [0.6, '#fee08b'],
  [0.7, '#fdae61'],
  [0.8, '#f46d43'],
  [0.9, '#d53e4f'],
  [1.0, '#9e0142'],
]

function xRange(barcodes: Bar[][], padLeft: number, padRight: number): [number, number] {
  let xmin = Infinity
  let xmax = -Infinity
  for (const bars of barcodes)
    for (const b of bars) {
      xmin = Math.min(xmin, b.birth)
      xmax = Math.max(xmax, b.death ?? b.birth)
    }
  const span = xmax - xmin || 1
  return [xmin - padLeft * span, xmax + padRight * span]
}

const PANEL_HEIGHT = 190

function monomialLabel(i: number): Partial<Layout>['annotations'] {
  return [
    {
      text: LABELS[i],
      xref: 'paper',
      yref: 'paper',
      x: 0.98,
      y: 0.96,
      showarrow: false,
      font: { size: 12 },
    },
  ]
}

// black border around the plot region
const BORDER: Partial<Layout>['shapes'] = [
  {
    type: 'rect',
    xref: 'paper',
    yref: 'paper',
    x0: 0,
    y0: 0,
    x1: 1,
    y1: 1,
    line: { color: 'black', width: 1 },
    layer: 'below',
  },
]

const TICKS = { ticks: 'outside', ticklen: 4, tickcolor: 'black' } as const

function baseLayout(i: number, xmin: number, xmax: number): Partial<Layout> {
  return {
    height: PANEL_HEIGHT,
    margin: { l: 40, r: 10, t: 6, b: 30 },
    xaxis: { range: [xmin, xmax], zeroline: false, ...TICKS },
    showlegend: false,
    annotations: monomialLabel(i),
    shapes: BORDER,
  }
}

// A genuinely square plot region with identical x/y ranges — equal aspect
// without scaleanchor (which would otherwise widen the x range to fill the
// panel width). Used by the diagram and image panels.
const SQ_MIN = 175 // 70% of the original 250px
const SQ_MAX = 360
const SQ_MARGIN = { l: 46, t: 8, b: 34 }

function squareLayout(i: number, xmin: number, xmax: number, size: number, rightMargin = 12): Partial<Layout> {
  return {
    width: SQ_MARGIN.l + size + rightMargin,
    height: SQ_MARGIN.t + size + SQ_MARGIN.b,
    margin: { l: SQ_MARGIN.l, r: rightMargin, t: SQ_MARGIN.t, b: SQ_MARGIN.b },
    xaxis: { range: [xmin, xmax], zeroline: false, ...TICKS },
    yaxis: { range: [xmin, xmax], zeroline: false, ...TICKS },
    showlegend: false,
    annotations: monomialLabel(i),
    shapes: BORDER,
  }
}

// Square size that makes the panel's d+1 plots fill the hosting panel's
// height (with a lower bound), and never overflow its width. `extraTop`
// accounts for content above the plots (e.g. the shared-range row).
function useSquareSize(nPlots: number, rightMargin: number, extraTop = 0) {
  const ref = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState(SQ_MIN)
  useEffect(() => {
    const holder = ref.current?.parentElement // the panel-content-holder
    if (!holder) return
    const recompute = () => {
      const availH = holder.clientHeight - extraTop - 10
      const availW = holder.clientWidth - SQ_MARGIN.l - rightMargin - 14
      const byHeight = Math.floor(availH / nPlots) - SQ_MARGIN.t - SQ_MARGIN.b
      const s = Math.min(byHeight, availW, SQ_MAX)
      setSize(Math.max(SQ_MIN, s))
    }
    recompute()
    const obs = new ResizeObserver(recompute)
    obs.observe(holder)
    return () => obs.disconnect()
  }, [nPlots, rightMargin, extraTop])
  return { ref, size }
}

function BarcodePlots({ results }: { results: ComputeResponse }) {
  const [xmin, xmax] = xRange(results.barcodes, 0.12, 0.05)
  // highest exponent on top, like plot_barcodes
  const dims = [...results.barcodes.keys()].reverse()
  return (
    <>
      {dims.map((i) => {
        const bars = results.barcodes[i]
        const traces: Data[] = bars.map((b, j) => ({
          x: [b.birth, b.death ?? xmax],
          y: [-j, -j],
          mode: 'lines',
          line: { color: 'black', width: 3 },
          hoverinfo: 'text',
          text: `[${b.birth.toFixed(3)}, ${b.death === null ? '∞' : b.death.toFixed(3)}] × ${b.multiplicity.toFixed(3)}`,
        }))
        traces.push({
          x: bars.map((b) => b.birth),
          y: bars.map((_, j) => -j),
          mode: 'text',
          text: bars.map((b) => b.multiplicity.toFixed(3)),
          textposition: 'middle left',
          textfont: { size: 9 },
          hoverinfo: 'skip',
        })
        // open endpoint where an infinite bar meets the border: [birth, ∞)
        const infBars = bars.map((b, j) => ({ b, j })).filter(({ b }) => b.death === null)
        if (infBars.length > 0) {
          traces.push({
            x: infBars.map(() => xmax),
            y: infBars.map(({ j }) => -j),
            mode: 'markers',
            marker: { symbol: 'circle', size: 7, color: 'white', line: { color: 'black', width: 1.5 } },
            cliponaxis: false,
            hoverinfo: 'skip',
          })
        }
        const layout = baseLayout(i, xmin, xmax)
        layout.yaxis = { visible: false, range: [-bars.length, 1] }
        return <Plot key={i} data={traces} layout={layout} config={{ displayModeBar: false }} useResizeHandler style={{ width: '100%' }} />
      })}
    </>
  )
}

function DiagramPlots({ results, size }: { results: ComputeResponse; size: number }) {
  const [xmin, xmax] = xRange(results.barcodes, 0.12, 0.12)
  const dims = [...results.barcodes.keys()].reverse()
  return (
    <>
      {dims.map((i) => {
        const bars = results.barcodes[i]
        const finite = bars.filter((b) => b.death !== null)
        const infinite = bars.filter((b) => b.death === null)
        const traces: Data[] = [
          {
            x: [xmin, xmax],
            y: [xmin, xmax],
            mode: 'lines',
            line: { color: 'gray', width: 1, dash: 'dash' },
            opacity: 0.4,
            hoverinfo: 'skip',
          },
          {
            x: finite.map((b) => b.birth),
            y: finite.map((b) => b.death as number),
            mode: 'markers+text',
            marker: { color: 'black', size: 5 },
            text: finite.map((b) => b.multiplicity.toFixed(3)),
            textposition: 'middle right',
            textfont: { size: 9 },
            hoverinfo: 'text',
            hovertext: finite.map(
              (b) => `(${b.birth.toFixed(3)}, ${(b.death as number).toFixed(3)}) × ${b.multiplicity.toFixed(3)}`,
            ),
          },
        ]
        if (infinite.length > 0) {
          // open marker pinned to the top border: death = ∞ (no multiplicity label)
          traces.push({
            x: infinite.map((b) => b.birth),
            y: infinite.map(() => xmax),
            mode: 'markers',
            marker: { symbol: 'circle', size: 7, color: 'white', line: { color: 'black', width: 1.5 } },
            cliponaxis: false,
            hoverinfo: 'text',
            hovertext: infinite.map((b) => `(${b.birth.toFixed(3)}, ∞) × ${b.multiplicity.toFixed(3)}`),
          })
        }
        const layout = squareLayout(i, xmin, xmax, size)
        return (
          <div key={i} className="square-plot">
            <Plot data={traces} layout={layout} config={{ displayModeBar: false }} />
          </div>
        )
      })}
    </>
  )
}

function ImagePlots({ results, sameRange, size }: { results: ComputeResponse; sameRange: boolean; size: number }) {
  const { xmin, xmax, data, size: gridSize } = results.images
  const axis = useMemo(
    () => Array.from({ length: gridSize }, (_, k) => xmin + ((k + 0.5) / gridSize) * (xmax - xmin)),
    [xmin, xmax, gridSize],
  )
  const globalRange = useMemo(() => {
    let m = 0
    for (const img of data) for (const row of img) for (const v of row) m = Math.max(m, Math.abs(v))
    return m || 1
  }, [data])
  const dims = [...data.keys()].reverse()
  return (
    <>
      {dims.map((i) => {
        let range = globalRange
        if (!sameRange) {
          range = 0
          for (const row of data[i]) for (const v of row) range = Math.max(range, Math.abs(v))
          range = range || 1
        }
        const traces: Data[] = [
          {
            type: 'heatmap',
            z: data[i],
            x: axis,
            y: axis,
            zmin: -range,
            zmax: range,
            colorscale: SPECTRAL_R,
            showscale: !sameRange,
          },
        ]
        const layout = squareLayout(i, xmin, xmax, size, sameRange ? 12 : 74)
        return (
          <div key={i} className="square-plot">
            <Plot data={traces} layout={layout} config={{ displayModeBar: false }} />
          </div>
        )
      })}
    </>
  )
}

// Standalone panel bodies for the floating-panel system.

export function BarcodePanel() {
  const results = useStore((s) => s.results)
  if (!results) return <div className="plots" />
  return (
    <div className="plots">
      <BarcodePlots results={results} />
    </div>
  )
}

export function DiagramPanel() {
  const results = useStore((s) => s.results)
  const { ref, size } = useSquareSize((results?.d ?? 2) + 1, 12)
  return (
    <div className="plots" ref={ref}>
      {results && <DiagramPlots results={results} size={size} />}
    </div>
  )
}

export function ImagePanel() {
  const results = useStore((s) => s.results)
  const sameRange = useStore((s) => s.ui.sameRange)
  const setUi = useStore((s) => s.setUi)
  const { ref, size } = useSquareSize((results?.d ?? 2) + 1, sameRange ? 12 : 74, 26)
  return (
    <div className="plots" ref={ref}>
      <label className="row image-options">
        <input type="checkbox" checked={sameRange} onChange={(e) => setUi({ sameRange: e.target.checked })} />
        shared range
      </label>
      {results && <ImagePlots results={results} sameRange={sameRange} size={size} />}
    </div>
  )
}
