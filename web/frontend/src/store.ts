import { create } from 'zustand'
import { compute, type ComputeResponse } from './api'
import { DEFAULT_PRESET, type Preset } from './presets'

export interface Inputs {
  d: 2 | 3
  lattice: number[][]
  points: number[][]
  weights: number[]
}

interface UiState {
  radius: number
  showPoints: boolean
  showBasis: boolean
  showDomains: boolean
  showFullSkeleton: boolean
  showArcs: boolean
  showFiltrationEdges: boolean
  showVoronoiFiltrationEdges: boolean
  showVoronoiSkeleton: boolean
  showVoronoiArcs: boolean
  showBalls: boolean
  ballOpacity: number
  sameRange: boolean
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
  setDimension: (d: 2 | 3) => void
}

let debounceTimer: ReturnType<typeof setTimeout> | undefined
let requestSeq = 0

function scheduleRecompute(get: () => State, set: (partial: Partial<State>) => void, delayMs = 300) {
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(async () => {
    const { inputs, ui } = get()
    const seq = ++requestSeq
    set({ status: 'loading' })
    try {
      const results = await compute({ ...inputs, imageSize: ui.imageSize })
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
    },
    results: null,
    status: 'idle',
    error: null,
    ui: {
      radius: 0,
      showPoints: true,
      showBasis: true,
      showDomains: true,
      showFullSkeleton: true,
      showArcs: true,
      showFiltrationEdges: true,
      showVoronoiFiltrationEdges: true,
      showVoronoiSkeleton: true,
      showVoronoiArcs: true,
      showBalls: true,
      ballOpacity: 0.35,
      sameRange: true,
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
      update(() => ({
        d: preset.d,
        lattice: preset.lattice.map((r) => [...r]),
        points: preset.points.map((r) => [...r]),
        weights: [...preset.weights],
      }), 0),

    setDimension: (d) => {
      if (d === get().inputs.d) return
      const identity = d === 2 ? [[1, 0], [0, 1]] : [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      update(() => ({
        d,
        lattice: identity,
        points: [new Array(d).fill(0.5)],
        weights: [0],
      }), 0)
    },
  }
})

// initial compute on module load
scheduleRecompute(useStore.getState as never, useStore.setState as never, 0)
