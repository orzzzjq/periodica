// Periodic marching cubes over a scalar field sampled on a uniform
// fractional grid (the 3D grid-input pipeline), plus the sublevel-set
// component labeling that links the extracted isosurface to the merge tree.
//
// The module is three.js-free and works on plain typed arrays so it can be
// smoke-tested from the console against `__store.getState().results.grid`.
//
// Grid conventions (matching periodica.core.periodic_grid): the field has
// shape [N1, N2, N3]; the quotient vertex id of grid index (i, j, k) is the
// C-order ravel i*N2*N3 + j*N3 + k; grid index (i, j, k) sits at fractional
// coordinates (i/N1, j/N2, k/N3), i.e. real position U · (idx / N).

// ---- standard marching cubes triangle table (Paul Bourke) -----------------
//
// Transcription validated: every row's edge set equals the independently
// derived crossing-edge set of its configuration, and randomized periodic
// fields (1000 runs) plus smooth-field topology checks (sphere chi=2,
// watertightness at saddle-heavy thresholds, wrap seams) found zero holes.
//
// Cube corners: c0..c3 the z=0 ring (0,0,0) (1,0,0) (1,1,0) (0,1,0), c4..c7
// the same ring at z=1. Edges 0-3 the bottom ring, 4-7 the top ring, 8-11
// the verticals c_i -> c_{i+4}. Corner bit i of the cube index is set iff
// f(c_i) <= threshold. Winding is irrelevant here: normals come from the
// field gradient and the surface is rendered double-sided.
const CORNER_OFFSETS: [number, number, number][] = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
]

const EDGE_CORNERS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]

// prettier-ignore
const TRI_TABLE: number[][] = [
  [], [0, 8, 3], [0, 1, 9], [1, 8, 3, 9, 8, 1], [1, 2, 10], [0, 8, 3, 1, 2, 10],
  [9, 2, 10, 0, 2, 9], [2, 8, 3, 2, 10, 8, 10, 9, 8], [3, 11, 2], [0, 11, 2, 8, 11, 0],
  [1, 9, 0, 2, 3, 11], [1, 11, 2, 1, 9, 11, 9, 8, 11], [3, 10, 1, 11, 10, 3],
  [0, 10, 1, 0, 8, 10, 8, 11, 10], [3, 9, 0, 3, 11, 9, 11, 10, 9], [9, 8, 10, 10, 8, 11],
  [4, 7, 8], [4, 3, 0, 7, 3, 4], [0, 1, 9, 8, 4, 7], [4, 1, 9, 4, 7, 1, 7, 3, 1],
  [1, 2, 10, 8, 4, 7], [3, 4, 7, 3, 0, 4, 1, 2, 10], [9, 2, 10, 9, 0, 2, 8, 4, 7],
  [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4], [8, 4, 7, 3, 11, 2], [11, 4, 7, 11, 2, 4, 2, 0, 4],
  [9, 0, 1, 8, 4, 7, 2, 3, 11], [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
  [3, 10, 1, 3, 11, 10, 7, 8, 4], [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
  [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3], [4, 7, 11, 4, 11, 9, 9, 11, 10],
  [9, 5, 4], [9, 5, 4, 0, 8, 3], [0, 5, 4, 1, 5, 0], [8, 5, 4, 8, 3, 5, 3, 1, 5],
  [1, 2, 10, 9, 5, 4], [3, 0, 8, 1, 2, 10, 4, 9, 5], [5, 2, 10, 5, 4, 2, 4, 0, 2],
  [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8], [9, 5, 4, 2, 3, 11], [0, 11, 2, 0, 8, 11, 4, 9, 5],
  [0, 5, 4, 0, 1, 5, 2, 3, 11], [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
  [10, 3, 11, 10, 1, 3, 9, 5, 4], [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
  [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3], [5, 4, 8, 5, 8, 10, 10, 8, 11],
  [9, 7, 8, 5, 7, 9], [9, 3, 0, 9, 5, 3, 5, 7, 3], [0, 7, 8, 0, 1, 7, 1, 5, 7],
  [1, 5, 3, 3, 5, 7], [9, 7, 8, 9, 5, 7, 10, 1, 2], [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
  [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2], [2, 10, 5, 2, 5, 3, 3, 5, 7],
  [7, 9, 5, 7, 8, 9, 3, 11, 2], [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
  [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7], [11, 2, 1, 11, 1, 7, 7, 1, 5],
  [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11], [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
  [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0], [11, 10, 5, 7, 11, 5],
  [10, 6, 5], [0, 8, 3, 5, 10, 6], [9, 0, 1, 5, 10, 6], [1, 8, 3, 1, 9, 8, 5, 10, 6],
  [1, 6, 5, 2, 6, 1], [1, 6, 5, 1, 2, 6, 3, 0, 8], [9, 6, 5, 9, 0, 6, 0, 2, 6],
  [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8], [2, 3, 11, 10, 6, 5], [11, 0, 8, 11, 2, 0, 10, 6, 5],
  [0, 1, 9, 2, 3, 11, 5, 10, 6], [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
  [6, 3, 11, 6, 5, 3, 5, 1, 3], [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
  [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9], [6, 5, 9, 6, 9, 11, 11, 9, 8],
  [5, 10, 6, 4, 7, 8], [4, 3, 0, 4, 7, 3, 6, 5, 10], [1, 9, 0, 5, 10, 6, 8, 4, 7],
  [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4], [6, 1, 2, 6, 5, 1, 4, 7, 8],
  [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7], [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
  [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9], [3, 11, 2, 7, 8, 4, 10, 6, 5],
  [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11], [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
  [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6], [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
  [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
  [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7], [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
  [10, 4, 9, 6, 4, 10], [4, 10, 6, 4, 9, 10, 0, 8, 3], [10, 0, 1, 10, 6, 0, 6, 4, 0],
  [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10], [1, 4, 9, 1, 2, 4, 2, 6, 4],
  [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4], [0, 2, 4, 4, 2, 6], [8, 3, 2, 8, 2, 4, 4, 2, 6],
  [10, 4, 9, 10, 6, 4, 11, 2, 3], [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
  [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10], [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
  [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3], [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
  [3, 11, 6, 3, 6, 0, 0, 6, 4], [6, 4, 8, 11, 6, 8],
  [7, 10, 6, 7, 8, 10, 8, 9, 10], [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
  [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0], [10, 6, 7, 10, 7, 1, 1, 7, 3],
  [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7], [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
  [7, 8, 0, 7, 0, 6, 6, 0, 2], [7, 3, 2, 6, 7, 2], [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
  [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
  [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11], [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
  [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6], [0, 9, 1, 11, 6, 7],
  [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0], [7, 11, 6],
  [7, 6, 11], [3, 0, 8, 11, 7, 6], [0, 1, 9, 11, 7, 6], [8, 1, 9, 8, 3, 1, 11, 7, 6],
  [10, 1, 2, 6, 11, 7], [1, 2, 10, 3, 0, 8, 6, 11, 7], [2, 9, 0, 2, 10, 9, 6, 11, 7],
  [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8], [7, 2, 3, 6, 2, 7], [7, 0, 8, 7, 6, 0, 6, 2, 0],
  [2, 7, 6, 2, 3, 7, 0, 1, 9], [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
  [10, 7, 6, 10, 1, 7, 1, 3, 7], [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
  [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7], [7, 6, 10, 7, 10, 8, 8, 10, 9],
  [6, 8, 4, 11, 8, 6], [3, 6, 11, 3, 0, 6, 0, 4, 6], [8, 6, 11, 8, 4, 6, 9, 0, 1],
  [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6], [6, 8, 4, 6, 11, 8, 2, 10, 1],
  [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6], [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
  [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3], [8, 2, 3, 8, 4, 2, 4, 6, 2],
  [0, 4, 2, 4, 6, 2], [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8], [1, 9, 4, 1, 4, 2, 2, 4, 6],
  [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1], [10, 1, 0, 10, 0, 6, 6, 0, 4],
  [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3], [10, 9, 4, 6, 10, 4],
  [4, 9, 5, 7, 6, 11], [0, 8, 3, 4, 9, 5, 11, 7, 6], [5, 0, 1, 5, 4, 0, 7, 6, 11],
  [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5], [9, 5, 4, 10, 1, 2, 7, 6, 11],
  [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5], [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
  [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6], [7, 2, 3, 7, 6, 2, 5, 4, 9],
  [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7], [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
  [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8], [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
  [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
  [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10], [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
  [6, 9, 5, 6, 11, 9, 11, 8, 9], [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
  [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11], [6, 11, 3, 6, 3, 5, 5, 3, 1],
  [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6], [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
  [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5], [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
  [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2], [9, 5, 6, 9, 6, 0, 0, 6, 2],
  [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8], [1, 5, 6, 2, 1, 6],
  [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6], [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
  [0, 3, 8, 5, 6, 10], [10, 5, 6],
  [11, 5, 10, 7, 5, 11], [11, 5, 10, 11, 7, 5, 8, 3, 0], [5, 11, 7, 5, 10, 11, 1, 9, 0],
  [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1], [11, 1, 2, 11, 7, 1, 7, 5, 1],
  [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11], [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
  [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2], [2, 5, 10, 2, 3, 5, 3, 7, 5],
  [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5], [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
  [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2], [1, 3, 5, 3, 7, 5],
  [0, 8, 7, 0, 7, 1, 1, 7, 5], [9, 0, 3, 9, 3, 5, 5, 3, 7], [9, 8, 7, 5, 9, 7],
  [5, 8, 4, 5, 10, 8, 10, 11, 8], [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
  [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5], [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
  [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8], [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
  [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5], [9, 4, 5, 2, 11, 3],
  [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4], [5, 10, 2, 5, 2, 4, 4, 2, 0],
  [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9], [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
  [8, 4, 5, 8, 5, 3, 3, 5, 1], [0, 4, 5, 1, 0, 5], [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
  [9, 4, 5],
  [4, 11, 7, 4, 9, 11, 9, 10, 11], [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
  [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11], [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
  [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2], [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
  [11, 7, 4, 11, 4, 2, 2, 4, 0], [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
  [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9], [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
  [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10], [1, 10, 2, 8, 7, 4],
  [4, 9, 1, 4, 1, 7, 7, 1, 3], [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1], [4, 0, 3, 7, 4, 3],
  [4, 8, 7],
  [9, 10, 8, 10, 11, 8], [3, 0, 9, 3, 9, 11, 11, 9, 10], [0, 1, 10, 0, 10, 8, 8, 10, 11],
  [3, 1, 10, 11, 3, 10], [1, 2, 11, 1, 11, 9, 9, 11, 8], [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
  [0, 2, 11, 8, 0, 11], [3, 2, 11], [2, 3, 8, 2, 8, 10, 10, 8, 9], [9, 10, 2, 0, 9, 2],
  [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8], [1, 10, 2], [1, 3, 8, 9, 1, 8], [0, 9, 1],
  [0, 3, 8], [],
]

// triangles per cube configuration, derived from TRI_TABLE at load
const TRI_COUNT = new Uint8Array(256)
for (let i = 0; i < 256; i++) TRI_COUNT[i] = TRI_TABLE[i].length / 3

// per-edge canonical endpoint order: (lo, hi) = ascending grid offset along
// the edge axis, so the interpolation is bit-identical for the same physical
// edge seen from any adjacent cell (crack-free, seamless under tiling)
const EDGE_LO = new Uint8Array(12)
const EDGE_HI = new Uint8Array(12)
const EDGE_AXIS = new Uint8Array(12)
for (let e = 0; e < 12; e++) {
  const [a, b] = EDGE_CORNERS[e]
  const oa = CORNER_OFFSETS[a]
  const ob = CORNER_OFFSETS[b]
  const axis = oa[0] !== ob[0] ? 0 : oa[1] !== ob[1] ? 1 : 2
  EDGE_AXIS[e] = axis
  EDGE_LO[e] = oa[axis] === 0 ? a : b
  EDGE_HI[e] = oa[axis] === 0 ? b : a
}

// ---- small linear algebra helpers -----------------------------------------

// 3x3 inverse via the adjugate; m is row-major number[][]
export function inverse3(m: number[][]): number[][] {
  const [a, b, c] = m[0]
  const [d, e, f] = m[1]
  const [g, h, i] = m[2]
  const A = e * i - f * h
  const B = f * g - d * i
  const C = d * h - e * g
  const det = a * A + b * B + c * C
  return [
    [A / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [C / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ]
}

// ---- field preparation (once per compute result) ---------------------------

export interface GridField {
  shape: [number, number, number]
  values: Float64Array // C-order (k fastest), the quotient vertex filtration
  // a_j = (column j of U) / N_j: real-space step per unit of grid index j
  cellCols: [number[], number[], number[]]
  // index-space gradient -> real-space gradient: U^{-T} · diag(N)
  gradXform: number[][]
}

export function buildGridField(shape: number[], values: number[], U: number[][]): GridField {
  const [N1, N2, N3] = shape
  const Uinv = inverse3(U)
  const cellCols: [number[], number[], number[]] = [
    [U[0][0] / N1, U[1][0] / N1, U[2][0] / N1],
    [U[0][1] / N2, U[1][1] / N2, U[2][1] / N2],
    [U[0][2] / N3, U[1][2] / N3, U[2][2] / N3],
  ]
  // gradXform[i][j] = (U^{-T})[i][j] * N_j = Uinv[j][i] * N_j
  const gradXform = [0, 1, 2].map((i) => [Uinv[0][i] * N1, Uinv[1][i] * N2, Uinv[2][i] * N3])
  return { shape: [N1, N2, N3], values: Float64Array.from(values), cellCols, gradXform }
}

// ---- marching cubes (once per threshold) ------------------------------------

export interface IsosurfaceMesh {
  positions: Float32Array // 9 floats per triangle (unindexed corners), real space
  normals: Float32Array // parallel to positions, unit, outward (+grad f)
  // per triangle: quotient id of the below-threshold endpoint of the grid
  // edge carrying the triangle's first vertex — lets the included/ghost
  // split rerun on a subtree-filter change without rerunning marching cubes
  triOwner: Uint32Array
  triangleCount: number
}

export function marchingCubes(field: GridField, threshold: number): IsosurfaceMesh {
  const [N1, N2, N3] = field.shape
  const f = field.values
  const [a1, a2, a3] = field.cellCols
  const gx = field.gradXform
  const nCells = N1 * N2 * N3
  const s1 = N2 * N3

  // pass 1: classify every cell, count triangles for exact allocation
  const cubeIdx = new Uint8Array(nCells)
  let nTri = 0
  {
    let cell = 0
    for (let i = 0; i < N1; i++) {
      const i1 = i + 1 === N1 ? 0 : i + 1
      for (let j = 0; j < N2; j++) {
        const j1 = j + 1 === N2 ? 0 : j + 1
        const r00 = i * s1 + j * N3
        const r10 = i1 * s1 + j * N3
        const r11 = i1 * s1 + j1 * N3
        const r01 = i * s1 + j1 * N3
        for (let k = 0; k < N3; k++, cell++) {
          const k1 = k + 1 === N3 ? 0 : k + 1
          let ci = 0
          if (f[r00 + k] <= threshold) ci |= 1
          if (f[r10 + k] <= threshold) ci |= 2
          if (f[r11 + k] <= threshold) ci |= 4
          if (f[r01 + k] <= threshold) ci |= 8
          if (f[r00 + k1] <= threshold) ci |= 16
          if (f[r10 + k1] <= threshold) ci |= 32
          if (f[r11 + k1] <= threshold) ci |= 64
          if (f[r01 + k1] <= threshold) ci |= 128
          cubeIdx[cell] = ci
          nTri += TRI_COUNT[ci]
        }
      }
    }
  }

  const positions = new Float32Array(nTri * 9)
  const normals = new Float32Array(nTri * 9)
  const triOwner = new Uint32Array(nTri)
  if (nTri === 0) return { positions, normals, triOwner, triangleCount: 0 }

  // wrapped ravel id and periodic central-difference gradient in index space
  const id = (i: number, j: number, k: number) =>
    ((i + N1) % N1) * s1 + ((j + N2) % N2) * N3 + ((k + N3) % N3)
  const grad = (i: number, j: number, k: number, out: number[]) => {
    const g1 = (f[id(i + 1, j, k)] - f[id(i - 1, j, k)]) / 2
    const g2 = (f[id(i, j + 1, k)] - f[id(i, j - 1, k)]) / 2
    const g3 = (f[id(i, j, k + 1)] - f[id(i, j, k - 1)]) / 2
    out[0] = g1
    out[1] = g2
    out[2] = g3
  }

  // pass 2: emit triangles, computing each active edge's crossing once per cell
  const exs = new Float64Array(12) // crossing positions (real space)
  const eys = new Float64Array(12)
  const ezs = new Float64Array(12)
  const enx = new Float64Array(12) // crossing normals
  const eny = new Float64Array(12)
  const enz = new Float64Array(12)
  const eBelow = new Uint32Array(12) // quotient id of the inside endpoint
  const gLo = [0, 0, 0]
  const gHi = [0, 0, 0]
  let out = 0 // float offset into positions/normals
  let tri = 0
  let cell = 0
  for (let i = 0; i < N1; i++) {
    for (let j = 0; j < N2; j++) {
      for (let k = 0; k < N3; k++, cell++) {
        const ci = cubeIdx[cell]
        if (ci === 0 || ci === 255) continue
        const tt = TRI_TABLE[ci]
        // the edges this configuration uses
        let edgeMask = 0
        for (let m = 0; m < tt.length; m++) edgeMask |= 1 << tt[m]
        for (let e = 0; e < 12; e++) {
          if (!(edgeMask & (1 << e))) continue
          const lo = EDGE_LO[e]
          const hi = EDGE_HI[e]
          const ol = CORNER_OFFSETS[lo]
          const oh = CORNER_OFFSETS[hi]
          const li = i + ol[0]
          const lj = j + ol[1]
          const lk = k + ol[2]
          const idLo = id(li, lj, lk)
          const idHi = id(i + oh[0], j + oh[1], k + oh[2])
          const fLo = f[idLo]
          const fHi = f[idHi]
          // exactly one endpoint is inside, so fLo !== fHi
          const s = (threshold - fLo) / (fHi - fLo)
          eBelow[e] = fLo <= threshold ? idLo : idHi
          // unwrapped real-valued grid coordinates (seamless across the wrap)
          const axis = EDGE_AXIS[e]
          const u = li + (axis === 0 ? s : 0)
          const v = lj + (axis === 1 ? s : 0)
          const w = lk + (axis === 2 ? s : 0)
          exs[e] = u * a1[0] + v * a2[0] + w * a3[0]
          eys[e] = u * a1[1] + v * a2[1] + w * a3[1]
          ezs[e] = u * a1[2] + v * a2[2] + w * a3[2]
          // gradient lerped along the edge, mapped to real space
          grad(li, lj, lk, gLo)
          grad(i + oh[0], j + oh[1], k + oh[2], gHi)
          const g1 = gLo[0] + s * (gHi[0] - gLo[0])
          const g2 = gLo[1] + s * (gHi[1] - gLo[1])
          const g3 = gLo[2] + s * (gHi[2] - gLo[2])
          let nx = gx[0][0] * g1 + gx[0][1] * g2 + gx[0][2] * g3
          let ny = gx[1][0] * g1 + gx[1][1] * g2 + gx[1][2] * g3
          let nz = gx[2][0] * g1 + gx[2][1] * g2 + gx[2][2] * g3
          const len = Math.hypot(nx, ny, nz)
          if (len > 1e-12) {
            nx /= len
            ny /= len
            nz /= len
          } else {
            // degenerate flat field: fall back to the edge axis
            nx = axis === 0 ? 1 : 0
            ny = axis === 1 ? 1 : 0
            nz = axis === 2 ? 1 : 0
          }
          enx[e] = nx
          eny[e] = ny
          enz[e] = nz
        }
        for (let m = 0; m < tt.length; m += 3, tri++) {
          triOwner[tri] = eBelow[tt[m]]
          for (let c = 0; c < 3; c++, out += 3) {
            const e = tt[m + c]
            positions[out] = exs[e]
            positions[out + 1] = eys[e]
            positions[out + 2] = ezs[e]
            normals[out] = enx[e]
            normals[out + 1] = eny[e]
            normals[out + 2] = enz[e]
          }
        }
      }
    }
  }

  return { positions, normals, triOwner, triangleCount: nTri }
}

// ---- sublevel-set component labeling (once per threshold) -------------------

// Union-find over the quotient vertices using the grid quotient arcs (the
// SAME adjacency the merge tree is built on, so the labeling matches the
// tree's components exactly). `sortedArcs` must be ascending by filtration.
// Returns fully path-compressed roots: roots[v] === roots[roots[v]].
export function labelSublevelComponents(
  sortedArcs: { filtration: number; vStart: number; vEnd: number }[],
  nVerts: number,
  threshold: number,
): Int32Array {
  const parent = new Int32Array(nVerts)
  for (let v = 0; v < nVerts; v++) parent[v] = v
  const size = new Int32Array(nVerts).fill(1)
  const find = (v: number) => {
    while (parent[v] !== v) {
      parent[v] = parent[parent[v]] // path halving
      v = parent[v]
    }
    return v
  }
  // binary search: arcs with filtration <= threshold form a prefix
  let lo = 0
  let hi = sortedArcs.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sortedArcs[mid].filtration <= threshold) lo = mid + 1
    else hi = mid
  }
  for (let m = 0; m < lo; m++) {
    const a = find(sortedArcs[m].vStart)
    const b = find(sortedArcs[m].vEnd)
    if (a === b) continue
    if (size[a] < size[b]) {
      parent[a] = b
      size[b] += size[a]
    } else {
      parent[b] = a
      size[a] += size[b]
    }
  }
  for (let v = 0; v < nVerts; v++) parent[v] = find(v)
  return parent
}

// ---- domain-boundary caps (solid sublevel sets) ------------------------------

// Trilinear sample of the periodic field at fractional coordinates (mod 1).
function trilinear(field: GridField, f1: number, f2: number, f3: number): number {
  const [N1, N2, N3] = field.shape
  const v = field.values
  const s1 = N2 * N3
  const u1 = (f1 - Math.floor(f1)) * N1
  const u2 = (f2 - Math.floor(f2)) * N2
  const u3 = (f3 - Math.floor(f3)) * N3
  const i = Math.floor(u1) % N1
  const j = Math.floor(u2) % N2
  const k = Math.floor(u3) % N3
  const a = u1 - Math.floor(u1)
  const b = u2 - Math.floor(u2)
  const c = u3 - Math.floor(u3)
  const i1 = i + 1 === N1 ? 0 : i + 1
  const j1 = j + 1 === N2 ? 0 : j + 1
  const k1 = k + 1 === N3 ? 0 : k + 1
  const c00 = v[i * s1 + j * N3 + k] * (1 - a) + v[i1 * s1 + j * N3 + k] * a
  const c10 = v[i * s1 + j1 * N3 + k] * (1 - a) + v[i1 * s1 + j1 * N3 + k] * a
  const c01 = v[i * s1 + j * N3 + k1] * (1 - a) + v[i1 * s1 + j * N3 + k1] * a
  const c11 = v[i * s1 + j1 * N3 + k1] * (1 - a) + v[i1 * s1 + j1 * N3 + k1] * a
  return (c00 * (1 - b) + c10 * b) * (1 - c) + (c01 * (1 - b) + c11 * b) * c
}

// Caps that close the sublevel solid where the 3x Dirichlet domain cuts it:
// on each domain facet, the filled region {f <= t} is triangulated by filled
// marching squares over a 2D grid on the facet plane (the periodic field is
// sampled trilinearly; the facet outline enters as the max with the other
// halfspace violations, so caps terminate at the polytope edges). The result
// is an IsosurfaceMesh in WORLD space (not instanced — the domain boundary
// is not periodic), with triOwner = nearest 3D grid node of each triangle's
// centroid so the component split works exactly like the surface's.
export function buildDomainCaps(
  field: GridField,
  U: number[][],
  domainA: number[][],
  domainB: number[],
  domainVertices: number[][],
  threshold: number,
): IsosurfaceMesh {
  const [N1, N2, N3] = field.shape
  const Uinv = inverse3(U)
  const frac = (x: number, y: number, z: number): [number, number, number] => [
    Uinv[0][0] * x + Uinv[0][1] * y + Uinv[0][2] * z,
    Uinv[1][0] * x + Uinv[1][1] * y + Uinv[1][2] * z,
    Uinv[2][0] * x + Uinv[2][1] * y + Uinv[2][2] * z,
  ]
  // 2D grid step matching the 3D sampling step (mismatched steps open
  // visible gaps where the cap meets the clipped surface)
  const h =
    (Math.hypot(U[0][0], U[1][0], U[2][0]) / N1 +
      Math.hypot(U[0][1], U[1][1], U[2][1]) / N2 +
      Math.hypot(U[0][2], U[1][2], U[2][2]) / N3) /
    3

  const positions: number[] = []
  const normals: number[] = []
  const owners: number[] = []
  const bScale = Math.max(1, Math.abs(Math.max(...domainB)))

  for (let fi = 0; fi < domainA.length; fi++) {
    const a = domainA[fi]
    const an = Math.hypot(a[0], a[1], a[2])
    const n = [a[0] / an, a[1] / an, a[2] / an]
    const dPlane = (3 * domainB[fi]) / an
    // facet polygon vertices: domain vertices lying on this plane
    const onPlane = domainVertices.filter(
      (v) => Math.abs(a[0] * v[0] + a[1] * v[1] + a[2] * v[2] - 3 * domainB[fi]) < 1e-6 * bScale * an,
    )
    if (onPlane.length < 3) continue
    // in-plane orthonormal basis (right-handed with n, so the walk below is
    // counterclockwise seen from outside)
    const ax = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]
    const e1raw = [
      ax[0] - (ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]) * n[0],
      ax[1] - (ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]) * n[1],
      ax[2] - (ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]) * n[2],
    ]
    const e1n = Math.hypot(e1raw[0], e1raw[1], e1raw[2])
    const e1 = [e1raw[0] / e1n, e1raw[1] / e1n, e1raw[2] / e1n]
    const e2 = [
      n[1] * e1[2] - n[2] * e1[1],
      n[2] * e1[0] - n[0] * e1[2],
      n[0] * e1[1] - n[1] * e1[0],
    ]
    const p0 = [n[0] * dPlane, n[1] * dPlane, n[2] * dPlane]
    let sMin = Infinity
    let sMax = -Infinity
    let tMin = Infinity
    let tMax = -Infinity
    for (const v of onPlane) {
      const dx = v[0] - p0[0]
      const dy = v[1] - p0[1]
      const dz = v[2] - p0[2]
      const s = dx * e1[0] + dy * e1[1] + dz * e1[2]
      const t = dx * e2[0] + dy * e2[1] + dz * e2[2]
      sMin = Math.min(sMin, s)
      sMax = Math.max(sMax, s)
      tMin = Math.min(tMin, t)
      tMax = Math.max(tMax, t)
    }
    sMin -= h
    sMax += h
    tMin -= h
    tMax += h
    const Ms = Math.min(160, Math.max(2, Math.ceil((sMax - sMin) / h)))
    const Mt = Math.min(160, Math.max(2, Math.ceil((tMax - tMin) / h)))
    const ds = (sMax - sMin) / Ms
    const dt = (tMax - tMin) / Mt

    // node values of g = max(f - threshold, other halfspace violations):
    // g <= 0 exactly on the solid's cross-section with this facet
    const gv = new Float64Array((Ms + 1) * (Mt + 1))
    const px = new Float64Array((Ms + 1) * (Mt + 1))
    const py = new Float64Array((Ms + 1) * (Mt + 1))
    const pz = new Float64Array((Ms + 1) * (Mt + 1))
    for (let is = 0; is <= Ms; is++) {
      const s = sMin + is * ds
      for (let it = 0; it <= Mt; it++) {
        const t = tMin + it * dt
        const x = p0[0] + s * e1[0] + t * e2[0]
        const y = p0[1] + s * e1[1] + t * e2[1]
        const z = p0[2] + s * e1[2] + t * e2[2]
        const [f1, f2, f3] = frac(x, y, z)
        let g = trilinear(field, f1, f2, f3) - threshold
        for (let j = 0; j < domainA.length; j++) {
          if (j === fi) continue
          const aj = domainA[j]
          const viol = aj[0] * x + aj[1] * y + aj[2] * z - 3 * domainB[j]
          if (viol > g) g = viol
        }
        const id = is * (Mt + 1) + it
        gv[id] = g
        px[id] = x
        py[id] = y
        pz[id] = z
      }
    }

    // filled marching squares: walk each cell boundary counterclockwise,
    // keeping inside corners and edge crossings; fan-triangulate the polygon
    const poly: number[][] = []
    for (let is = 0; is < Ms; is++) {
      for (let it = 0; it < Mt; it++) {
        const c = [
          is * (Mt + 1) + it,
          (is + 1) * (Mt + 1) + it,
          (is + 1) * (Mt + 1) + it + 1,
          is * (Mt + 1) + it + 1,
        ]
        poly.length = 0
        for (let e = 0; e < 4; e++) {
          const ia = c[e]
          const ib = c[(e + 1) % 4]
          const ga = gv[ia]
          const gb = gv[ib]
          if (ga <= 0) poly.push([px[ia], py[ia], pz[ia]])
          if ((ga <= 0) !== (gb <= 0)) {
            const w = ga / (ga - gb)
            poly.push([
              px[ia] + w * (px[ib] - px[ia]),
              py[ia] + w * (py[ib] - py[ia]),
              pz[ia] + w * (pz[ib] - pz[ia]),
            ])
          }
        }
        if (poly.length < 3) continue
        for (let v = 1; v + 1 < poly.length; v++) {
          const tri = [poly[0], poly[v], poly[v + 1]]
          const cx = (tri[0][0] + tri[1][0] + tri[2][0]) / 3
          const cy = (tri[0][1] + tri[1][1] + tri[2][1]) / 3
          const cz = (tri[0][2] + tri[1][2] + tri[2][2]) / 3
          const [f1, f2, f3] = frac(cx, cy, cz)
          const g1 = ((Math.round((f1 - Math.floor(f1)) * N1) % N1) + N1) % N1
          const g2 = ((Math.round((f2 - Math.floor(f2)) * N2) % N2) + N2) % N2
          const g3 = ((Math.round((f3 - Math.floor(f3)) * N3) % N3) + N3) % N3
          owners.push(g1 * N2 * N3 + g2 * N3 + g3)
          for (const p of tri) {
            positions.push(p[0], p[1], p[2])
            normals.push(n[0], n[1], n[2]) // outward: the solid lies behind
          }
        }
      }
    }
  }

  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    triOwner: Uint32Array.from(owners),
    triangleCount: owners.length,
  }
}

// ---- included/ghost split (the only step rerun on a filter change) ----------

// Splits the mesh triangles by whether their component (the sublevel
// component of the triangle's below-threshold grid vertex) contains a
// subtree-filtered vertex. Returns corner-index buffers for
// BufferGeometry.setIndex over the mesh's shared position/normal attributes;
// includedIndex === null means "no filter: use the unindexed mesh as-is".
//
// Known v1 limitation: marching cubes resolves ambiguous faces independently
// of the arc-graph connectivity, so very near a saddle threshold a patch can
// transiently attribute to a neighboring component — cosmetic only.
export function splitTriangles(
  mesh: IsosurfaceMesh,
  roots: Int32Array,
  filtered: Set<number> | null,
): { includedIndex: Uint32Array | null; ghostIndex: Uint32Array } {
  if (!filtered) return { includedIndex: null, ghostIndex: new Uint32Array(0) }
  const includedRoots = new Set<number>()
  for (const v of filtered) if (v >= 0 && v < roots.length) includedRoots.add(roots[v])
  const n = mesh.triangleCount
  let nIn = 0
  for (let t = 0; t < n; t++) if (includedRoots.has(roots[mesh.triOwner[t]])) nIn++
  const includedIndex = new Uint32Array(nIn * 3)
  const ghostIndex = new Uint32Array((n - nIn) * 3)
  let a = 0
  let b = 0
  for (let t = 0; t < n; t++) {
    if (includedRoots.has(roots[mesh.triOwner[t]])) {
      includedIndex[a++] = 3 * t
      includedIndex[a++] = 3 * t + 1
      includedIndex[a++] = 3 * t + 2
    } else {
      ghostIndex[b++] = 3 * t
      ghostIndex[b++] = 3 * t + 1
      ghostIndex[b++] = 3 * t + 2
    }
  }
  return { includedIndex, ghostIndex }
}
