import { create } from 'zustand'
import { compute, type ComputeResponse } from './api'
import { DEFAULT_PRESET, type Preset } from './presets'

export interface TreeViewState {
  x: [number, number]
  y: [number, number]
  // zoom anchor: time t on branch row; sliderT is the filtration slider value
  // synced on the click (t - 1e-6 for merge corners, t + 1e-6 for events)
  sub?: { row: number; t: number; sliderT: number }
}

export interface Inputs {
  d: 2 | 3
  lattice: number[][]
  points: number[][]
  weights: number[]
  // how the point coordinates are interpreted: 'fractional' = coefficients
  // of the lattice basis vectors (converted to real before the API call),
  // 'real' = Cartesian coordinates as-is
  coordMode: 'fractional' | 'real'
}

interface UiState {
  radius: number // f_Del: Delaunay filtration threshold (power-distance scale)
  radiusVor: number // f_Vor: Voronoi filtration threshold (negated power-distance scale)
  showPoints: boolean
  showBasis: boolean
  showDomains: boolean
  showFullSkeleton: boolean
  showArcs: boolean
  showFiltrationEdges: boolean
  showVoronoiFiltrationEdges: boolean
  showVoronoiPoints: boolean
  showVoronoiSkeleton: boolean
  showVoronoiArcs: boolean
  showBalls: boolean
  showVoronoiBalls: boolean
  ballOpacity: number // Delaunay filtration balls
  filtEdgeOpacity: number // Delaunay filtration edges
  coneOpacity: number // Voronoi filtration cones
  vorEdgeOpacity: number // Voronoi filtration edges
  sameRange: boolean
  showTreeMultiplicity: boolean // monomial labels on the merge tree
  // null = full view; sub = the zoom anchor (time t on branch row): the
  // part of the tree not flowing into that point is dimmed
  treeView: TreeViewState | null
  // history of views left by subtree clicks (null = full view), for the
  // back button; manual drag-zooms replace the current view without pushing
  treeViewStack: (TreeViewState | null)[]
  // while a subtree view is active: the quotient vertices of the clicked
  // connected component — the matching complex's filtration overlays show
  // only edges/cones with both endpoints (and balls with their point) in it
  subtreeFilter: { complex: 'delaunay' | 'voronoi'; verts: number[] } | null
  imageSize: number
  complexType: 'delaunay' | 'voronoi'
}

interface State {
  inputs: Inputs
  results: ComputeResponse | null
  status: 'idle' | 'loading' | 'error'
  error: string | null
  ui: UiState
  setUi: (partial: Partial<UiState>) => void
  setLatticeEntry: (i: number, j: number, value: number) => void
  setPointCoord: (row: number, j: number, value: number) => void
  setWeight: (row: number, value: number) => void
  addPoint: () => void
  removePoint: (row: number) => void
  applyPreset: (preset: Preset) => void
  applyRandom: (seed: number, nPoints: number) => void
  setDimension: (d: 2 | 3) => void
  setCoordMode: (mode: 'fractional' | 'real') => void
}

let debounceTimer: ReturnType<typeof setTimeout> | undefined
let requestSeq = 0

// deterministic 32-bit PRNG (mulberry32) for the reproducible Random preset
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Random geometry: lattice entries uniform in [-1, 1] (redrawn until the
// basis is not near-singular), fractional points uniform in [0, 1], all
// rounded to 3 decimals for readable inputs. The lattice is drawn before
// the points, so with a fixed seed a larger point count keeps the lattice
// and the existing points and only appends new ones.
function randomGeometry(d: 2 | 3, seed: number, nPoints: number) {
  const rng = mulberry32(seed)
  const r3 = (x: number) => Math.round(x * 1000) / 1000
  const det = (m: number[][]) =>
    d === 2
      ? m[0][0] * m[1][1] - m[0][1] * m[1][0]
      : m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  let lattice: number[][] = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)),
  )
  for (let tries = 0; tries < 100; tries++) {
    const cand = Array.from({ length: d }, () =>
      Array.from({ length: d }, () => r3(rng() * 2 - 1)),
    )
    if (Math.abs(det(cand)) > (d === 2 ? 0.3 : 0.15)) {
      lattice = cand
      break
    }
  }
  const points = Array.from({ length: nPoints }, () =>
    Array.from({ length: d }, () => r3(rng())),
  )
  return { lattice, points, weights: new Array(nPoints).fill(0) }
}

// fractional -> real: point coordinates are coefficients of the basis
// vectors (lattice columns are the vectors), real_i = sum_j U[i][j] * p_j
function toRealPoints(inputs: Inputs): number[][] {
  if (inputs.coordMode !== 'fractional') return inputs.points
  return inputs.points.map((p) =>
    inputs.lattice.map((row) => row.reduce((s, u, j) => s + u * (p[j] ?? 0), 0)),
  )
}

function scheduleRecompute(get: () => State, set: (partial: Partial<State>) => void, delayMs = 300) {
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(async () => {
    const { inputs, ui } = get()
    const seq = ++requestSeq
    set({ status: 'loading' })
    try {
      const results = await compute({
        d: inputs.d,
        lattice: inputs.lattice,
        points: toRealPoints(inputs),
        weights: inputs.weights,
        imageSize: ui.imageSize,
      })
      if (seq !== requestSeq) return // a newer request superseded this one
      set({ results, status: 'idle', error: null })
    } catch (e) {
      if (seq !== requestSeq) return
      set({ status: 'error', error: e instanceof Error ? e.message : String(e) })
    }
  }, delayMs)
}

export const useStore = create<State>((set, get) => {
  const update = (mutate: (inputs: Inputs) => Inputs, delayMs = 300) => {
    set({ inputs: mutate(get().inputs) })
    scheduleRecompute(get, set, delayMs)
  }

  return {
    inputs: {
      d: DEFAULT_PRESET.d,
      lattice: DEFAULT_PRESET.lattice.map((r) => [...r]),
      points: DEFAULT_PRESET.points.map((r) => [...r]),
      weights: [...DEFAULT_PRESET.weights],
      coordMode: 'fractional',
    },
    results: null,
    status: 'idle',
    error: null,
    ui: {
      radius: 0,
      // -Infinity = "at the slider minimum" (the actual minimum depends on
      // the computed Voronoi barcode); the UI clamps it into range
      radiusVor: -Infinity,
      showPoints: true,
      showBasis: true,
      showDomains: true,
      showFullSkeleton: true,
      showArcs: true,
      showFiltrationEdges: true,
      showVoronoiFiltrationEdges: true,
      showVoronoiPoints: true,
      showVoronoiSkeleton: true,
      showVoronoiArcs: true,
      showBalls: true,
      showVoronoiBalls: true,
      ballOpacity: 0.35,
      filtEdgeOpacity: 1,
      coneOpacity: 0.35,
      vorEdgeOpacity: 1,
      sameRange: true,
      showTreeMultiplicity: true,
      treeView: null,
      treeViewStack: [],
      subtreeFilter: null,
      imageSize: 100,
      complexType: 'delaunay',
    },

    setUi: (partial) => {
      const prev = get().ui
      set({ ui: { ...prev, ...partial } })
      if (partial.imageSize !== undefined && partial.imageSize !== prev.imageSize) {
        scheduleRecompute(get, set)
      }
    },

    setLatticeEntry: (i, j, value) =>
      update((inp) => {
        const lattice = inp.lattice.map((r) => [...r])
        lattice[i][j] = value
        return { ...inp, lattice }
      }),

    setPointCoord: (row, j, value) =>
      update((inp) => {
        const points = inp.points.map((r) => [...r])
        points[row][j] = value
        return { ...inp, points }
      }),

    setWeight: (row, value) =>
      update((inp) => {
        const weights = [...inp.weights]
        weights[row] = value
        return { ...inp, weights }
      }),

    addPoint: () =>
      update((inp) => ({
        ...inp,
        points: [...inp.points, new Array(inp.d).fill(0.5)],
        weights: [...inp.weights, 0],
      }), 0),

    removePoint: (row) =>
      update((inp) => ({
        ...inp,
        points: inp.points.filter((_, i) => i !== row),
        weights: inp.weights.filter((_, i) => i !== row),
      }), 0),

    applyPreset: (preset) =>
      update((inp) => ({
        ...inp,
        d: preset.d,
        lattice: preset.lattice.map((r) => [...r]),
        points: preset.points.map((r) => [...r]),
        weights: [...preset.weights],
      }), 0),

    applyRandom: (seed, nPoints) =>
      update((inp) => {
        const n = Math.max(1, Math.min(100, Math.round(nPoints) || 1))
        // random points are fractional by construction
        return { ...inp, ...randomGeometry(inp.d, seed, n), coordMode: 'fractional' as const }
      }, 0),

    setDimension: (d) => {
      if (d === get().inputs.d) return
      const identity = d === 2 ? [[1, 0], [0, 1]] : [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      update((inp) => ({
        ...inp,
        d,
        lattice: identity,
        points: [new Array(d).fill(0.5)],
        weights: [0],
      }), 0)
    },

    setCoordMode: (mode) => {
      if (mode === get().inputs.coordMode) return
      // the entered numbers are kept and reinterpreted in the new mode
      update((inp) => ({ ...inp, coordMode: mode }), 0)
    },
  }
})

// initial compute on module load
scheduleRecompute(useStore.getState as never, useStore.setState as never, 0)

// debugging/testing probe
if (typeof window !== 'undefined') {
  ;(window as unknown as { __store: typeof useStore }).__store = useStore
}
