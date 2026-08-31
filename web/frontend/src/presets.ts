export interface Preset {
  name: string
  d: 2 | 3
  lattice: number[][]
  points: number[][]
  weights: number[]
}

// Ported from the EXAMPLES in main.py, plus a hidden-point demo.
export const PRESETS: Preset[] = [
  {
    name: '2D square',
    d: 2,
    lattice: [
      [1, 0],
      [0, 1],
    ],
    points: [[0.5, 0.5]],
    weights: [0],
  },
  {
    name: '2D hexagonal',
    d: 2,
    lattice: [
      [1, 0.5],
      [0, Math.sqrt(3) / 2],
    ],
    points: [[0, 0]],
    weights: [0],
  },
  {
    name: '2D weighted',
    d: 2,
    lattice: [
      [1, 0],
      [0, 1],
    ],
    points: [
      [0, 0],
      [0.5, 0],
      [0, 0.5],
      [0.5, 0.5],
    ],
    weights: [0.04, 0.01, 0.01, 0.02],
  },
  {
    name: '2D hidden point',
    d: 2,
    lattice: [
      [1, 0],
      [0, 1],
    ],
    points: [
      [0.3, 0.25],
      [0.25, 0.25],
    ],
    weights: [0, 0.09],
  },
  {
    name: '3D cubic',
    d: 3,
    lattice: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    points: [[0.5, 0.5, 0.5]],
    weights: [0],
  },
  {
    name: '3D two points',
    d: 3,
    lattice: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    points: [
      [0.25, 0.25, 0.25],
      [0.75, 0.75, 0.75],
    ],
    weights: [0, 0],
  },
]

export const DEFAULT_PRESET = PRESETS[0]
