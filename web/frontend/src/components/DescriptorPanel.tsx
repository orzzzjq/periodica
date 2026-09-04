import { useEffect, useMemo, useRef, useState } from 'react'
import Plotly from 'plotly.js-dist-min'
import type { Data, Layout } from 'plotly.js'
import createPlotlyComponent from 'react-plotly.js/factory'
import type { Bar, Descriptors, ImagesData, TreeEvent } from '../api'
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

// exported: the radius sliders reuse the barcode plot's x-range so their
// bounds match the plotted xmax exactly
export function xRange(barcodes: Bar[][], padLeft: number, padRight: number): [number, number] {
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

// One barcode plot: 154px plot region + the same t/b margins as the square
// diagram/image plots, so all descriptors share the same vertical rhythm.
const PANEL_HEIGHT = 196

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

function baseLayout(i: number, xmin: number, xmax: number, width: number): Partial<Layout> {
  return {
    width,
    height: PANEL_HEIGHT,
    margin: { l: 46, r: 46, t: 8, b: 34 },
    xaxis: { range: [xmin, xmax], zeroline: false, ...TICKS },
    showlegend: false,
    annotations: monomialLabel(i),
    shapes: BORDER,
  }
}

// Width that tracks the hosting panel. Explicitly measured (rather than
// Plotly's useResizeHandler, which relies on window resize events and goes
// stale while the panel is minimized into the app bar): the ResizeObserver
// re-fires when the content holder is reattached to the DOM on restore.
function usePanelWidth(minW = 300) {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(minW)
  const [height, setHeight] = useState(200)
  useEffect(() => {
    const holder = ref.current?.parentElement // the panel-content-holder
    if (!holder) return
    const recompute = () => {
      if (holder.clientWidth > 0) setWidth(Math.max(minW, holder.clientWidth - 18))
      if (holder.clientHeight > 0) setHeight(Math.max(140, holder.clientHeight - 14))
    }
    recompute()
    const obs = new ResizeObserver(recompute)
    obs.observe(holder)
    return () => obs.disconnect()
  }, [minW])
  return { ref, width, height }
}

// A genuinely square plot region with identical x/y ranges — equal aspect
// without scaleanchor (which would otherwise widen the x range to fill the
// panel width). Used by the diagram and image panels.
const SQ_MIN = PANEL_HEIGHT - 8 - 34 // matches the barcode plot-region height (154px)
const SQ_MAX = 360
// Room for the always-on colorbar: bar + pads + worst-case tick labels.
// Must be generous enough that Plotly never auto-expands the margin (which
// would shrink the plot area and break the square shape).
const IMG_RIGHT_MARGIN = 80
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

function BarcodePlots({
  barcodes,
  width,
  radius,
  cursorColor,
}: {
  barcodes: Bar[][]
  width: number
  radius: number | null
  cursorColor: string
}) {
  const [xmin, xmax] = xRange(barcodes, 0.12, 0.05)
  // live filtration cursor linked to the visualization slider (null = hidden)
  const radiusLine: Partial<Layout>['shapes'] =
    radius !== null
      ? [
          {
            type: 'line',
            x0: radius,
            x1: radius,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: cursorColor, width: 1, dash: 'dash' },
          },
        ]
      : []
  // highest exponent on top, like plot_barcodes
  const dims = [...barcodes.keys()].reverse()
  return (
    <>
      {dims.map((i) => {
        const bars = barcodes[i]
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
        const layout = baseLayout(i, xmin, xmax, width)
        layout.yaxis = { visible: false, range: [-bars.length, 1] }
        layout.shapes = [...(layout.shapes ?? []), ...radiusLine]
        return <Plot key={i} data={traces} layout={layout} config={{ displayModeBar: false }} />
      })}
    </>
  )
}

function DiagramPlots({
  barcodes,
  size,
  cursor,
  cursorColor,
}: {
  barcodes: Bar[][]
  size: number
  cursor: number | null
  cursorColor: string
}) {
  const [xmin, xmax] = xRange(barcodes, 0.12, 0.12)
  const dims = [...barcodes.keys()].reverse()
  return (
    <>
      {dims.map((i) => {
        const bars = barcodes[i]
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
        layout.shapes = [...(layout.shapes ?? []), ...cursorSegments(cursor, cursorColor, xmin, xmax)]
        return (
          <div key={i} className="square-plot">
            <Plot data={traces} layout={layout} config={{ displayModeBar: false }} />
          </div>
        )
      })}
    </>
  )
}

function ImagePlots({
  images,
  sameRange,
  size,
  cursor,
  cursorColor,
}: {
  images: ImagesData
  sameRange: boolean
  size: number
  cursor: number | null
  cursorColor: string
}) {
  const { xmin, xmax, data, size: gridSize } = images
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
            showscale: true,
            colorbar: {
              thickness: 10,
              lenmode: 'pixels',
              len: size, // exactly the square plot-region height
              ypad: 0, // default 10px padding would shrink the bar inside len
              xpad: 10,
              yanchor: 'middle',
              y: 0.5, // centered on the plot region (paper coords)
              outlinecolor: 'black',
              outlinewidth: 1,
              ticks: 'outside',
              ticklen: 3,
              tickfont: { size: 9 },
            },
          },
        ]
        const layout = squareLayout(i, xmin, xmax, size, IMG_RIGHT_MARGIN)
        layout.shapes = [...(layout.shapes ?? []), ...cursorSegments(cursor, cursorColor, xmin, xmax)]
        return (
          <div key={i} className="square-plot">
            <Plot data={traces} layout={layout} config={{ displayModeBar: false }} />
          </div>
        )
      })}
    </>
  )
}

// Standalone panel bodies for the floating-panel system. The app-bar
// Delaunay/Voronoi toggle selects which complex's descriptors they show.

function useDescriptors(): { desc: Descriptors | null; error: string | null } {
  const results = useStore((s) => s.results)
  const which = useStore((s) => s.ui.complexType)
  if (!results) return { desc: null, error: null }
  if (which === 'voronoi') {
    return { desc: results.voronoi, error: results.voronoi ? null : (results.voronoiError ?? 'Voronoi unavailable') }
  }
  return { desc: { barcodes: results.barcodes, images: results.images, tree: results.tree }, error: null }
}

function DescError({ error }: { error: string | null }) {
  return error ? <div className="desc-error">Voronoi computation failed: {error}</div> : null
}

// Filtration cursor for the active complex: f_Del (blue) or f_Vor (red),
// hidden (null) while its slider sits at either end of its range (the
// barcode's minimum birth, or the slider maximum = the barcode plot xmax).
function useFiltrationCursor(desc: Descriptors | null): { cursor: number | null; cursorColor: string } {
  const radius = useStore((s) => s.ui.radius)
  const radiusVor = useStore((s) => s.ui.radiusVor)
  const which = useStore((s) => s.ui.complexType)
  const minB = desc ? xRange(desc.barcodes, 0, 0)[0] : -Infinity
  const maxB = desc ? xRange(desc.barcodes, 0.12, 0.05)[1] : Infinity // = slider max
  const visible = (v: number) => Number.isFinite(v) && v > minB + 1e-9 && v < maxB - 1e-9
  const value = which === 'voronoi' ? radiusVor : radius
  return { cursor: visible(value) ? value : null, cursorColor: which === 'voronoi' ? 'red' : 'blue' }
}

// The diagram/image cursor: for threshold f, the vertical segment x=f with
// y from f to ymax and the horizontal segment y=f with x from xmin to f —
// the boundary of the region of (birth, death) pairs alive at f.
function cursorSegments(
  cursor: number | null,
  cursorColor: string,
  xmin: number,
  xmax: number,
): NonNullable<Partial<Layout>['shapes']> {
  if (cursor === null) return []
  const line = { color: cursorColor, width: 1, dash: 'dash' as const }
  return [
    { type: 'line', x0: cursor, x1: cursor, y0: cursor, y1: xmax, line },
    { type: 'line', x0: xmin, x1: cursor, y0: cursor, y1: cursor, line },
  ]
}

export function BarcodePanel() {
  const { desc, error } = useDescriptors()
  const { ref, width } = usePanelWidth()
  const { cursor, cursorColor } = useFiltrationCursor(desc)
  return (
    <div className="plots" ref={ref}>
      <DescError error={error} />
      {desc && <BarcodePlots barcodes={desc.barcodes} width={width} radius={cursor} cursorColor={cursorColor} />}
    </div>
  )
}

export function DiagramPanel() {
  const results = useStore((s) => s.results)
  const { desc, error } = useDescriptors()
  const { ref, size } = useSquareSize((results?.d ?? 2) + 1, 12)
  const { cursor, cursorColor } = useFiltrationCursor(desc)
  return (
    <div className="plots" ref={ref}>
      <DescError error={error} />
      {desc && <DiagramPlots barcodes={desc.barcodes} size={size} cursor={cursor} cursorColor={cursorColor} />}
    </div>
  )
}

export function ImagePanel() {
  const results = useStore((s) => s.results)
  const { desc, error } = useDescriptors()
  const sameRange = useStore((s) => s.ui.sameRange)
  const { ref, size } = useSquareSize((results?.d ?? 2) + 1, IMG_RIGHT_MARGIN)
  const { cursor, cursorColor } = useFiltrationCursor(desc)
  return (
    <div className="plots" ref={ref}>
      <DescError error={error} />
      {desc && (
        <ImagePlots images={desc.images} sameRange={sameRange} size={size} cursor={cursor} cursorColor={cursorColor} />
      )}
    </div>
  )
}

// ---- merge tree ----

const SUPERSCRIPTS = ['⁰', '¹', '²', '³']

// pretty-print a shadow-monomial coefficient, recognizing k·π and k·π/3
function fmtCoeff(c: number): string {
  for (let k = 1; k <= 12; k++) {
    if (Math.abs(c - k * Math.PI) < 1e-9) return k === 1 ? 'π' : `${k}π`
  }
  for (let k = 1; k <= 12; k++) {
    if (k % 3 !== 0 && Math.abs(c - (k * Math.PI) / 3) < 1e-9) return `${k}π/3`
  }
  if (Math.abs(c - Math.round(c)) < 1e-9) return String(Math.round(c))
  return String(parseFloat(c.toPrecision(3)))
}

// volume of the k-dimensional unit ball: the shadow monomial of a component
// with multiplicity m and wrap dimension k is m·vol_k·R^k
const BALL_VOL = [1, 2, Math.PI, (4 * Math.PI) / 3]

function monomialText(coeff: number, exp: number): string {
  const total = coeff * BALL_VOL[exp]
  if (exp === 0) return fmtCoeff(total)
  const co = fmtCoeff(total)
  return `${co === '1' ? '' : co}R${exp === 1 ? '' : SUPERSCRIPTS[exp]}`
}

interface TreeBranch {
  birth: number
  death: number | null // null = essential branch, extends to xmax
  row: number
  parentRow: number | null
  subtreeRows: number // the subtree occupies rows [row, row + subtreeRows)
  subtreeMinBirth: number // earliest birth in the subtree
  events: { t: number; label: string }[] // monomial events (ticks + labels)
}

// Left-to-right layout, root at the bottom. Recursive placement: a branch
// takes the next row, then its children (sorted by merge time, earliest
// first) stack their subtrees above it. Every branch in an earlier-merging
// subtree is dead before a later sibling's merge time, so the later
// sibling's vertical connector crosses no live line.
function layoutTree(tree: TreeEvent[][]): TreeBranch[] {
  const n = tree.length
  const parent = new Array<number>(n).fill(-1)
  const death = new Array<number | null>(n).fill(null)
  const children: { id: number; t: number }[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < n; k++) {
    for (const [t, , , child] of tree[k]) {
      if (child !== -1 && child !== k && t !== null) {
        parent[child] = k
        death[child] = t
        children[k].push({ id: child, t })
      }
    }
  }
  const rows = new Array<number>(n).fill(0)
  const subRows = new Array<number>(n).fill(1)
  const subMin = new Array<number>(n).fill(Infinity)
  let next = 0
  const place = (i: number) => {
    rows[i] = next++
    children[i].sort((a, b) => a.t - b.t)
    let mb = tree[i][0]?.[0] ?? 0
    for (const c of children[i]) {
      place(c.id)
      mb = Math.min(mb, subMin[c.id])
    }
    subRows[i] = next - rows[i]
    subMin[i] = mb
  }
  const roots = [...Array(n).keys()].filter((i) => parent[i] === -1)
  roots.sort((a, b) => (tree[a][0]?.[0] ?? 0) - (tree[b][0]?.[0] ?? 0))
  for (const r of roots) place(r)

  const branches: TreeBranch[] = []
  for (let i = 0; i < n; i++) {
    const beam = tree[i]
    if (beam.length === 0) continue
    const raw: { t: number; coeff: number; exp: number }[] = []
    for (const [t, coeff, exp, child] of beam) {
      if (child === i || t === null) continue // own death: monomial unchanged
      raw.push({ t, coeff, exp })
    }
    // Several monomial changes can happen at the same time (up to fp noise);
    // keep only the final state: the smallest exponent, then coefficient.
    const events: { t: number; label: string }[] = []
    const eps = (t: number) => 1e-7 * Math.max(1, Math.abs(t))
    let cluster: typeof raw = []
    const flush = () => {
      if (cluster.length === 0) return
      let best = cluster[0]
      for (const e of cluster) {
        if (e.exp < best.exp || (e.exp === best.exp && e.coeff < best.coeff)) best = e
      }
      events.push({ t: cluster[cluster.length - 1].t, label: monomialText(best.coeff, best.exp) })
      cluster = []
    }
    for (const e of raw) {
      if (cluster.length > 0 && e.t - cluster[cluster.length - 1].t > eps(e.t)) flush()
      cluster.push(e)
    }
    flush()
    branches.push({
      birth: beam[0][0] ?? 0,
      death: death[i],
      row: rows[i],
      parentRow: parent[i] === -1 ? null : rows[parent[i]],
      subtreeRows: subRows[i],
      subtreeMinBirth: subMin[i],
      events,
    })
  }
  return branches
}

function MergeTreePlot({
  tree,
  barcodes,
  width,
  height,
  cursor,
  cursorColor,
  showLabels,
}: {
  tree: TreeEvent[][]
  barcodes: Bar[][]
  width: number
  height: number
  cursor: number | null
  cursorColor: string
  showLabels: boolean
}) {
  const [xmin, xmax] = xRange(barcodes, 0.12, 0.05)
  const branches = useMemo(() => layoutTree(tree), [tree])
  const maxRow = branches.reduce((m, b) => Math.max(m, b.row), 0)

  // interactive view: null = full tree; clicking a hover point zooms to the
  // subtree rooted there (pushing the previous view for the back button);
  // manual drag-zooms are kept via onRelayout
  const view = useStore((s) => s.ui.treeView)
  const stack = useStore((s) => s.ui.treeViewStack)
  const setUi = useStore((s) => s.setUi)
  const rowToBranch = useMemo(() => new Map(branches.map((b) => [b.row, b])), [branches])

  const zoomToSubtree = (row: number, t: number) => {
    const b = rowToBranch.get(row)
    if (!b) return
    // subtree bounding rectangle with symmetric padding, clamped to the
    // default (full-view) axis ranges
    const w = t - b.subtreeMinBirth || (xmax - xmin) * 0.01
    const rTop = b.row + b.subtreeRows - 1
    const padY = Math.max(0.08 * (rTop - b.row), 0.9)
    setUi({
      treeView: {
        x: [Math.max(xmin, b.subtreeMinBirth - 0.08 * w), Math.min(xmax, t + 0.08 * w)],
        y: [Math.max(-0.9, b.row - padY), Math.min(maxRow + 1.1, rTop + padY)],
        sub: { row: b.row, t },
      },
      treeViewStack: [...stack, view],
    })
  }

  // In a subtree view anchored at time t on branch b0, everything NOT
  // flowing into that point is dimmed to 25% opacity: branches outside b0's
  // subtree, descendants that merge into b0 only after t, b0's own
  // segments and events past t, and b0's merge connector.
  const sub = view?.sub
  const anchor = sub ? rowToBranch.get(sub.row) : undefined
  const included = useMemo(() => {
    const set = new Set<number>()
    if (!sub || !anchor) return set
    const childrenByRow = new Map<number, TreeBranch[]>()
    for (const b of branches) {
      if (b.parentRow === null) continue
      const arr = childrenByRow.get(b.parentRow) ?? []
      arr.push(b)
      childrenByRow.set(b.parentRow, arr)
    }
    const eps = 1e-9 * Math.max(1, Math.abs(sub.t))
    // children of the anchor that merged by time t, with their full subtrees
    const stack = (childrenByRow.get(anchor.row) ?? []).filter(
      (c) => c.death !== null && c.death <= sub.t + eps,
    )
    while (stack.length) {
      const b = stack.pop()!
      set.add(b.row)
      for (const c of childrenByRow.get(b.row) ?? []) stack.push(c)
    }
    return set
  }, [branches, sub, anchor])

  interface DrawGroup {
    lx: (number | null)[]
    ly: (number | null)[]
    ex: number[]
    ey: number[]
    etext: string[]
  }
  const newGroup = (): DrawGroup => ({ lx: [], ly: [], ex: [], ey: [], etext: [] })
  const main = newGroup()
  const faded = newGroup()
  const seg = (g: DrawGroup, x0: number, x1: number, row: number) => {
    g.lx.push(x0, x1, null)
    g.ly.push(row, row, null)
  }
  const drop = (g: DrawGroup, x: number, r0: number, r1: number) => {
    g.lx.push(x, x, null)
    g.ly.push(r0, r1, null)
  }
  const evt = (g: DrawGroup, t: number, row: number, label: string) => {
    g.ex.push(t)
    g.ey.push(row)
    g.etext.push(label)
  }

  for (const b of branches) {
    const end = b.death ?? xmax
    let lineTo: DrawGroup
    if (!sub || !anchor) {
      lineTo = main
    } else if (b.row === anchor.row) {
      // the anchor branch: opaque up to t, dimmed past it (incl. connector)
      const eps = 1e-9 * Math.max(1, Math.abs(sub.t))
      const tcut = Math.min(sub.t, end)
      seg(main, b.birth, tcut, b.row)
      if (end > tcut) seg(faded, tcut, end, b.row)
      if (b.parentRow !== null && b.death !== null) drop(faded, b.death, b.row, b.parentRow)
      for (const ev of b.events) evt(ev.t <= sub.t + eps ? main : faded, ev.t, b.row, ev.label)
      continue
    } else {
      lineTo = included.has(b.row) ? main : faded
    }
    seg(lineTo, b.birth, end, b.row)
    if (b.parentRow !== null && b.death !== null) drop(lineTo, b.death, b.row, b.parentRow)
    for (const ev of b.events) evt(lineTo, ev.t, b.row, ev.label)
  }

  const groupTraces = (g: DrawGroup, opacity: number): Data[] => [
    { x: g.lx, y: g.ly, mode: 'lines', line: { color: 'black', width: 2.5 }, opacity, hoverinfo: 'skip' },
    {
      x: g.ex,
      y: g.ey,
      mode: 'markers',
      marker: { symbol: 'line-ns-open', size: 7, color: 'black', line: { width: 1.5 } },
      opacity,
      hoverinfo: 'text',
      hovertext: g.ex.map((t, i) => `${g.etext[i]} (${t.toFixed(3)})`),
    },
    ...(showLabels
      ? [
          // top left: the label ends just left of the event, so merge
          // connectors (vertical lines at the event x) never cover it
          {
            x: g.ex,
            y: g.ey,
            mode: 'text',
            text: g.etext,
            textposition: 'top left',
            textfont: { size: 12 },
            opacity,
            cliponaxis: false,
            hoverinfo: 'skip',
          } as Data,
        ]
      : []),
  ]

  // invisible hover targets at the merge corners (where a branch turns
  // down into its parent) — over all branches, so a dimmed corner can
  // still be clicked to zoom to its subtree
  const mx: number[] = []
  const my: number[] = []
  for (const b of branches) {
    if (b.death !== null && b.parentRow !== null) {
      mx.push(b.death)
      my.push(b.row)
    }
  }

  const traces: Data[] = [
    ...groupTraces(faded, 0.25),
    ...groupTraces(main, 1),
    {
      x: mx,
      y: my,
      mode: 'markers',
      marker: { size: 10, color: 'rgba(0,0,0,0)' },
      hoverinfo: 'text',
      hovertext: mx.map((t) => `merge (${t.toFixed(3)})`),
      // the tooltip background inherits the (transparent) marker color;
      // pin it to the same opaque style as the event tooltips
      hoverlabel: { bgcolor: 'black', font: { color: 'white' } },
    },
  ]

  const cursorLine: NonNullable<Partial<Layout>['shapes']> =
    cursor !== null
      ? [
          {
            type: 'line',
            x0: cursor,
            x1: cursor,
            y0: 0,
            y1: 1,
            yref: 'paper',
            line: { color: cursorColor, width: 1, dash: 'dash' },
          },
        ]
      : []

  // fill the panel height (so top/bottom whitespace stays symmetric), but
  // never squeeze rows below readability
  const plotHeight = Math.max(height, 42 + (maxRow + 1) * 20 + 16)
  const layout: Partial<Layout> = {
    width,
    height: plotHeight,
    margin: { l: 46, r: 46, t: 8, b: 34 },
    xaxis: { range: view?.x ?? [xmin, xmax], zeroline: false, ...TICKS },
    yaxis: { visible: false, range: view?.y ?? [-0.9, maxRow + 1.1] },
    showlegend: false,
    shapes: [...(BORDER ?? []), ...cursorLine],
  }
  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ displayModeBar: false }}
      onClick={(e) => {
        const p = e.points?.[0]
        if (!p || typeof p.x !== 'number' || typeof p.y !== 'number') return
        zoomToSubtree(p.y, p.x)
      }}
      onRelayout={(e) => {
        if (!e) return
        if (e['xaxis.autorange'] || e['yaxis.autorange']) {
          setUi({ treeView: null })
          return
        }
        const x0 = e['xaxis.range[0]'] as number | undefined
        const x1 = e['xaxis.range[1]'] as number | undefined
        if (x0 === undefined || x1 === undefined) return
        const y0 = (e['yaxis.range[0]'] as number | undefined) ?? view?.y[0] ?? -0.9
        const y1 = (e['yaxis.range[1]'] as number | undefined) ?? view?.y[1] ?? maxRow + 1.1
        setUi({ treeView: { x: [x0, x1], y: [y0, y1], sub: view?.sub } })
      }}
    />
  )
}

export function MergeTreePanel() {
  const { desc, error } = useDescriptors()
  const { ref, width, height } = usePanelWidth()
  const { cursor, cursorColor } = useFiltrationCursor(desc)
  const showLabels = useStore((s) => s.ui.showTreeMultiplicity)
  const setUi = useStore((s) => s.setUi)
  const stack = useStore((s) => s.ui.treeViewStack)
  // a new tree (recompute, Delaunay/Voronoi switch) invalidates the zoom view
  const tree = desc?.tree
  useEffect(() => {
    setUi({ treeView: null, treeViewStack: [] })
  }, [tree, setUi])
  const goBack = () => {
    if (stack.length === 0) return
    setUi({ treeView: stack[stack.length - 1], treeViewStack: stack.slice(0, -1) })
  }
  return (
    <div className="plots" ref={ref}>
      <DescError error={error} />
      {stack.length > 0 && (
        <button className="tree-back" title="back to previous view" onClick={goBack}>
          <svg width="17" height="17" viewBox="0 0 24 24">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z" fill="currentColor" />
          </svg>
        </button>
      )}
      <button
        className="tree-reset"
        title="reset view"
        onClick={() => setUi({ treeView: null, treeViewStack: [] })}
      >
        <svg width="17" height="17" viewBox="0 0 24 24">
          <path d="M12 5V1L7 6l5 5V7a5 5 0 1 1-5 5H5a7 7 0 1 0 7-7z" fill="currentColor" />
        </svg>
      </button>
      {desc?.tree && (
        <MergeTreePlot
          tree={desc.tree}
          barcodes={desc.barcodes}
          width={width}
          height={height}
          cursor={cursor}
          cursorColor={cursorColor}
          showLabels={showLabels}
        />
      )}
    </div>
  )
}

// display popup for the merge tree panel header
export function TreeDisplayOptions() {
  const show = useStore((s) => s.ui.showTreeMultiplicity)
  const setUi = useStore((s) => s.setUi)
  const [open, setOpen] = useState(false)
  return (
    <div className="popup-control">
      <button className={open ? 'active' : ''} onClick={() => setOpen(!open)}>
        display {open ? '▴' : '▾'}
      </button>
      {open && (
        <div className="popup-panel">
          <label className="row">
            <input
              type="checkbox"
              checked={show}
              onChange={(e) => setUi({ showTreeMultiplicity: e.target.checked })}
            />
            Multiplicity
          </label>
        </div>
      )}
    </div>
  )
}

// Lives in the panel header's tab bar while the image tab is active.
export function SharedRangeToggle() {
  const sameRange = useStore((s) => s.ui.sameRange)
  const setUi = useStore((s) => s.setUi)
  return (
    <label className="row">
      <input type="checkbox" checked={sameRange} onChange={(e) => setUi({ sameRange: e.target.checked })} />
      shared scale
    </label>
  )
}
