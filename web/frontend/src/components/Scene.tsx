import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, MapControls, OrbitControls, OrthographicCamera } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Line2 } from 'three-stdlib'
import { inDirichletDomain, type ComputeResponse, type Polytope2D, type Polytope3D } from '../api'
import { captureRegistry } from '../capture'
import {
  buildDomainCaps,
  buildGridField,
  inverse3,
  labelSublevelComponents,
  marchingCubes,
  splitTriangles,
  type IsosurfaceMesh,
} from '../marchingCubes'
import {
  buildCapPieces,
  buildPowerCells,
  buildPowerCells2D,
  buildRegionPieces2D,
  powerCaps,
  powerRegion2D,
  powerSurface,
  surfaceTranslates,
  type CapPieces,
  type PowerCells,
  type PowerCells2D,
} from '../powerSurface'
import { useStore } from '../store'

const GREEN = '#30b830'
const BLUE = '#0000fe'
const RED = '#dd2222'
const FILL = '#fbe5d6'

function to3(p: number[]): [number, number, number] {
  return [p[0], p[1], p[2] ?? 0]
}

function BasisArrows({ basis }: { basis: number[][] }) {
  return (
    <>
      {basis.map((v, i) => {
        const dir = new THREE.Vector3(...to3(v))
        const length = dir.length()
        return (
          <arrowHelper
            key={i}
            args={[dir.clone().normalize(), new THREE.Vector3(0, 0, 0), length, GREEN, 0.08, 0.05]}
          />
        )
      })}
    </>
  )
}

function Domain2D({ polytope, fill, z }: { polytope: Polytope2D; fill: boolean; z: number }) {
  const loop = polytope.outline
  const shape = useMemo(() => {
    const s = new THREE.Shape()
    s.moveTo(loop[0][0], loop[0][1])
    for (let i = 1; i < loop.length; i++) s.lineTo(loop[i][0], loop[i][1])
    s.closePath()
    return s
  }, [loop])
  const outline = useMemo(
    () => [...loop, loop[0]].map((p) => new THREE.Vector3(p[0], p[1], z)),
    [loop, z],
  )
  return (
    <>
      {fill && (
        <mesh position={[0, 0, z - 0.001]}>
          <shapeGeometry args={[shape]} />
          <meshBasicMaterial color={FILL} />
        </mesh>
      )}
      <Line points={outline} color="black" lineWidth={fill ? 1.5 : 1} />
    </>
  )
}

function Domain3D({ polytope, translucent }: { polytope: Polytope3D; translucent: boolean }) {
  const { vertices, edges, triangles } = polytope
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3))
    g.setIndex(triangles.flat())
    g.computeVertexNormals()
    return g
  }, [vertices, triangles])
  return (
    <>
      {translucent && (
        <mesh geometry={geometry}>
          <meshBasicMaterial color="black" transparent opacity={0.06} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {edges.map(([a, b], i) => (
        <Line key={i} points={[to3(vertices[a]), to3(vertices[b])]} color="black" lineWidth={1} transparent opacity={0.6} />
      ))}
    </>
  )
}

function Points({ results }: { results: ComputeResponse }) {
  const { positions3x, originalIndex, canonicalCount, hidden } = results.points
  const hiddenSet = useMemo(() => new Set(hidden), [hidden])
  return (
    <>
      {positions3x.map((p, i) => {
        const orig = originalIndex[i]
        const isCanonical = i < canonicalCount
        const isHidden = hiddenSet.has(orig)
        // canonical points dark blue, periodic copies light blue
        const color = isHidden ? '#bbbbbb' : isCanonical ? '#00008b' : '#6f95d8'
        const r = isCanonical ? 0.035 : 0.022
        return (
          <mesh key={i} position={to3(p)}>
            <sphereGeometry args={[r, 16, 16]} />
            <meshBasicMaterial color={color} />
          </mesh>
        )
      })}
    </>
  )
}

function FullSkeleton({ results }: { results: ComputeResponse }) {
  const geometry = useMemo(() => {
    const pos: number[] = []
    for (const [s, t] of results.fullEdges) {
      pos.push(...to3(results.points.positions3x[s]), ...to3(results.points.positions3x[t]))
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    return g
  }, [results])
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={BLUE} transparent opacity={0.35} />
    </lineSegments>
  )
}

function QuotientArcs({ results }: { results: ComputeResponse }) {
  return (
    <>
      {results.quotientArcs.map((arc, i) => (
        <Line key={i} points={[to3(arc.start), to3(arc.end)]} color={BLUE} lineWidth={2.5} />
      ))}
    </>
  )
}

// Replicate resolved arc segments by lattice shifts z in [-3,3]^d, keeping
// copies with both endpoints inside the 3x Dirichlet domain. BFS from the
// canonical copy (z=0): by convexity a direction is never extended past its
// first out-of-domain copy.
function tile3xSegments(
  arcs: { start: number[]; end: number[] }[],
  results: ComputeResponse,
  zLift: number,
): [number, number, number][] {
  const { d, basis, domainA, domainB } = results
  const pts: [number, number, number][] = []
  for (const arc of arcs) {
    const seen = new Set<string>()
    const queue: number[][] = [new Array(d).fill(0)]
    seen.add(queue[0].join(','))
    while (queue.length) {
      const z = queue.pop()!
      const t = [0, 0, 0]
      for (let k = 0; k < d; k++) for (let j = 0; j < d; j++) t[j] += z[k] * basis[k][j]
      const a = [arc.start[0] + t[0], arc.start[1] + t[1], (arc.start[2] ?? 0) + t[2]]
      const b = [arc.end[0] + t[0], arc.end[1] + t[1], (arc.end[2] ?? 0) + t[2]]
      if (!inDirichletDomain(a, domainA, domainB) || !inDirichletDomain(b, domainA, domainB))
        continue
      pts.push([a[0], a[1], a[2] + zLift], [b[0], b[1], b[2] + zLift])
      for (let k = 0; k < d; k++)
        for (const step of [1, -1]) {
          const nz = [...z]
          nz[k] += step
          if (Math.abs(nz[k]) > 3) continue
          const key = nz.join(',')
          if (!seen.has(key)) {
            seen.add(key)
            queue.push(nz)
          }
        }
    }
  }
  return pts
}

// tolerance: the slider bounds come from the barcodes, which match edge
// filtration values only up to floating-point rounding
const filtEps = (radius: number) => 1e-9 * Math.max(1, Math.abs(radius))

// Quotient vertices of the connected component picked in the merge tree's
// subtree view (null = no restriction). Only the filtration overlays of the
// complex the zoomed merge tree belongs to are filtered.
function useSubtreeVerts(complex: 'delaunay' | 'voronoi'): Set<number> | null {
  const filter = useStore((s) => s.ui.subtreeFilter)
  return useMemo(
    () => (filter && filter.complex === complex ? new Set(filter.verts) : null),
    [filter, complex],
  )
}

// All tiled copies of all arcs, ordered by arc filtration value: the
// sublevel set at any threshold f is a prefix of the segment list. Built
// once per compute result — the slider never re-tiles.
interface TiledSegments {
  points: [number, number, number][] // two entries per segment
  filtration: number[] // per segment, ascending
}

function buildTiledSegments(
  arcs: { start: number[]; end: number[]; filtration: number }[],
  results: ComputeResponse,
  zLift: number,
): TiledSegments {
  const sorted = [...arcs].sort((a, b) => a.filtration - b.filtration)
  const points: [number, number, number][] = []
  const filtration: number[] = []
  for (const arc of sorted) {
    const segs = tile3xSegments([arc], results, zLift)
    for (const p of segs) points.push(p)
    for (let i = 0; i < segs.length / 2; i++) filtration.push(arc.filtration)
  }
  return { points, filtration }
}

// Draws the first `count` segments of a prefix-sorted TiledSegments buffer.
// The full geometry lives on the GPU once (LineSegments2 is instanced);
// each slider tick only updates instanceCount — zero re-upload.
function PrefixSegments({
  data,
  threshold,
  color,
  opacity,
}: {
  data: TiledSegments
  threshold: number
  color: string
  opacity: number
}) {
  const ref = useRef<Line2>(null)

  // binary search: number of segments with filtration <= threshold
  const t = threshold + filtEps(threshold)
  let lo = 0
  let hi = data.filtration.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (data.filtration[mid] <= t) lo = mid + 1
    else hi = mid
  }
  const count = lo

  // set every frame: robust against drei rebuilding the geometry internally
  useFrame(() => {
    if (ref.current) ref.current.geometry.instanceCount = count
  })

  if (data.points.length === 0) return null
  return (
    <Line
      ref={ref}
      points={data.points}
      segments
      color={color}
      lineWidth={5}
      transparent
      opacity={opacity}
      visible={count > 0}
    />
  )
}

// Sublevel set of the Delaunay filtration: the periodic edges whose
// power-scale filtration value is below the current threshold f_Del
// (slider-linked), tiled across the 3x Dirichlet domain.
function FiltrationEdges({ results, radius }: { results: ComputeResponse; radius: number }) {
  const verts = useSubtreeVerts('delaunay')
  const data = useMemo(() => {
    const arcs = verts
      ? results.quotientArcs.filter((a) => verts.has(a.vStart) && verts.has(a.vEnd))
      : results.quotientArcs
    return buildTiledSegments(arcs, results, results.d === 2 ? 0.002 : 0)
  }, [results, verts])
  const opacity = useStore((s) => s.ui.filtEdgeOpacity)
  return <PrefixSegments data={data} threshold={radius} color={BLUE} opacity={opacity} />
}

// ---- grid mode: colored points + nearest-neighbor edges of a scalar field ----

// compact viridis approximation, v normalized to [0, 1]
const VIRIDIS: [number, number, number][] = [
  [0.267, 0.005, 0.329],
  [0.229, 0.322, 0.545],
  [0.128, 0.567, 0.551],
  [0.369, 0.789, 0.383],
  [0.993, 0.906, 0.144],
]

function valueColor(v: number, min: number, max: number): string {
  const t = max > min ? (v - min) / (max - min) : 0.5
  const x = Math.min(Math.max(t, 0), 1) * (VIRIDIS.length - 1)
  const i = Math.min(Math.floor(x), VIRIDIS.length - 2)
  const f = x - i
  const c = VIRIDIS[i].map((a, k) => a + f * (VIRIDIS[i + 1][k] - a))
  return new THREE.Color(c[0], c[1], c[2]).getStyle()
}

// Grid points colored by function value; points not yet in the sublevel set
// (value > f) are ghosted, as are points outside a picked merge-tree subtree.
//
// Two instanced meshes sharing one instance ordering (ascending by value,
// subtree-excluded points last): the opaque mesh draws the first k born
// instances, the ghost mesh holds the same instances REVERSED and draws the
// remaining n-k — a slider tick only changes two instance counts, mirroring
// PrefixSegments (fine grids have ~10^5 copies in the 3x domain).
function GridPoints({ results, radius }: { results: ComputeResponse; radius: number }) {
  const g = results.grid
  const verts = useSubtreeVerts('delaunay')
  const bornRef = useRef<THREE.InstancedMesh>(null)
  const ghostRef = useRef<THREE.InstancedMesh>(null)

  const built = useMemo(() => {
    if (!g) return null
    const vmin = Math.min(...g.values)
    const vmax = Math.max(...g.values)
    // born threshold key: excluded-from-subtree points never turn opaque
    const key = (i: number) => {
      const orig = g.originalIndex[i]
      return verts && !verts.has(orig) ? Infinity : g.values[orig]
    }
    const order = Array.from({ length: g.positions3x.length }, (_, i) => i)
    order.sort((a, b) => key(a) - key(b))
    return { order, sortedKeys: order.map(key), vmin, vmax }
  }, [g, verts])

  // fill both instance buffers once per (results, subtree) change
  useEffect(() => {
    if (!built || !g || !bornRef.current || !ghostRef.current) return
    const m = new THREE.Matrix4()
    const color = new THREE.Color()
    const n = built.order.length
    for (let k = 0; k < n; k++) {
      const i = built.order[k]
      const p = g.positions3x[i]
      const s = g.canonical[i] ? 0.035 : 0.022
      m.makeScale(s, s, s).setPosition(p[0], p[1], p[2] ?? 0)
      color.set(valueColor(g.values[g.originalIndex[i]], built.vmin, built.vmax))
      bornRef.current.setMatrixAt(k, m)
      bornRef.current.setColorAt(k, color)
      ghostRef.current.setMatrixAt(n - 1 - k, m)
      ghostRef.current.setColorAt(n - 1 - k, color)
    }
    for (const ref of [bornRef, ghostRef]) {
      ref.current!.instanceMatrix.needsUpdate = true
      ref.current!.instanceColor!.needsUpdate = true
    }
  }, [built, g])

  // binary search: instances with value <= threshold are born
  const t = radius + filtEps(radius)
  let lo = 0
  let hi = built ? built.sortedKeys.length : 0
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (built!.sortedKeys[mid] <= t) lo = mid + 1
    else hi = mid
  }
  const bornCount = lo

  const n = built ? built.order.length : 0
  useFrame(() => {
    if (bornRef.current) bornRef.current.count = bornCount
    if (ghostRef.current) ghostRef.current.count = n - bornCount
  })

  if (!built || !g || n === 0) return null
  // fine grids have ~10^5 instances: flat discs in the 2D ortho view and a
  // low-poly sphere in 3D keep the triangle count in the low millions
  const dot = results.d === 2 ? <circleGeometry args={[1, 12]} /> : <sphereGeometry args={[1, 8, 6]} />
  return (
    // per-instance positions break the default bounding-sphere culling
    <>
      <instancedMesh key={`born-${n}`} ref={bornRef} args={[undefined, undefined, n]} frustumCulled={false}>
        {dot}
        <meshBasicMaterial />
      </instancedMesh>
      <instancedMesh key={`ghost-${n}`} ref={ghostRef} args={[undefined, undefined, n]} frustumCulled={false}>
        {dot}
        <meshBasicMaterial transparent opacity={0.2} depthWrite={false} />
      </instancedMesh>
    </>
  )
}

// All grid edges (the d nearest-neighbor directions), tiled over the 3x
// domain — the grid analogue of the full Delaunay skeleton.
function GridSkeleton({ results }: { results: ComputeResponse }) {
  const geometry = useMemo(() => {
    if (!results.grid) return null
    const pts = tile3xSegments(results.grid.arcs, results, results.d === 2 ? 0.001 : 0)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pts.flat(), 3))
    return g
  }, [results])
  if (!geometry) return null
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={BLUE} transparent opacity={0.35} />
    </lineSegments>
  )
}

// Sublevel set of the lower-star filtration: grid edges with
// max(f_start, f_end) below the threshold, tiled over the 3x domain.
// With negate, the superlevel counterpart: edges of the NEGATED field with
// max(-f_start, -f_end) below the f_Sup threshold (the -f scale), red like
// the Voronoi overlays and filtered by the superlevel subtree channel.
function GridFiltrationEdges({
  results,
  radius,
  negate = false,
}: {
  results: ComputeResponse
  radius: number
  negate?: boolean
}) {
  const verts = useSubtreeVerts(negate ? 'voronoi' : 'delaunay')
  const data = useMemo(() => {
    const g = results.grid
    if (!g) return { points: [], filtration: [] } as TiledSegments
    let arcs = negate
      ? // negate BEFORE the lower-star max: max(-f_u, -f_v) = -min(f_u, f_v)
        g.arcs.map((a) => ({
          ...a,
          filtration: Math.max(-g.values[a.vStart], -g.values[a.vEnd]),
        }))
      : g.arcs
    if (verts) arcs = arcs.filter((a) => verts.has(a.vStart) && verts.has(a.vEnd))
    return buildTiledSegments(arcs, results, results.d === 2 ? (negate ? 0.005 : 0.002) : 0)
  }, [results, verts, negate])
  const opacity = useStore((s) => (negate ? s.ui.vorEdgeOpacity : s.ui.filtEdgeOpacity))
  return (
    <PrefixSegments data={data} threshold={radius} color={negate ? RED : BLUE} opacity={opacity} />
  )
}

// isosurface colors match the Delaunay/Voronoi convention: sublevel blue,
// superlevel red
const ISO_COLOR = '#8fb0e8'
const ISO_COLOR_SUP = '#e08f8f'

// Integer combos z of the reduced-basis rows whose translated U-parallelepiped
// can intersect the 3x Dirichlet domain. Per-halfspace test with the exact
// support h_i of the centered cell (min over the cell of A_i·x = A_i·c − h_i),
// conservative for the intersection — excess copies are cut away by the
// clipping planes. The translates generate the same lattice as U's columns,
// so enumerating reduced-basis combos covers every cell copy. 3D only.
function latticeTranslates3x(results: ComputeResponse, U: number[][]): [number, number, number][] {
  const { basis, domainA, domainB } = results
  // cell center c0 = U·(½,½,½) and support along each domain normal:
  // h_i = ½ Σ_j |A_i · (column j of U)|
  const c0 = [0, 1, 2].map((r) => 0.5 * (U[r][0] + U[r][1] + U[r][2]))
  const support = domainA.map(
    (a) =>
      0.5 *
      (Math.abs(a[0] * U[0][0] + a[1] * U[1][0] + a[2] * U[2][0]) +
        Math.abs(a[0] * U[0][1] + a[1] * U[1][1] + a[2] * U[2][1]) +
        Math.abs(a[0] * U[0][2] + a[1] * U[1][2] + a[2] * U[2][2])),
  )
  // enumeration box: |z_k| from the bounding-sphere reach (loose is fine here)
  const rCell = Math.hypot(
    Math.abs(U[0][0]) + Math.abs(U[0][1]) + Math.abs(U[0][2]),
    Math.abs(U[1][0]) + Math.abs(U[1][1]) + Math.abs(U[1][2]),
    Math.abs(U[2][0]) + Math.abs(U[2][1]) + Math.abs(U[2][2]),
  )
  let R = 0
  for (const v of results.domain3x.vertices) R = Math.max(R, Math.hypot(v[0], v[1], v[2] ?? 0))
  const reach = R + rCell + Math.hypot(c0[0], c0[1], c0[2])
  // translate T = Bᵀz (B rows = basis vectors) ⇒ z = B⁻ᵀT, |z_k| ≤ |col_k(B⁻¹)|·|T|
  const Binv = inverse3(basis)
  const zMax = [0, 1, 2].map((k) =>
    Math.ceil(Math.hypot(Binv[0][k], Binv[1][k], Binv[2][k]) * reach),
  )
  const out: [number, number, number][] = []
  for (let z1 = -zMax[0]; z1 <= zMax[0]; z1++)
    for (let z2 = -zMax[1]; z2 <= zMax[1]; z2++)
      for (let z3 = -zMax[2]; z3 <= zMax[2]; z3++) {
        const cx = c0[0] + z1 * basis[0][0] + z2 * basis[1][0] + z3 * basis[2][0]
        const cy = c0[1] + z1 * basis[0][1] + z2 * basis[1][1] + z3 * basis[2][1]
        const cz = c0[2] + z1 * basis[0][2] + z2 * basis[1][2] + z3 * basis[2][2]
        let keep = true
        for (let i = 0; i < domainA.length && keep; i++) {
          const a = domainA[i]
          if (a[0] * cx + a[1] * cy + a[2] * cz > 3 * domainB[i] + support[i]) keep = false
        }
        if (keep) out.push([cx - c0[0], cy - c0[1], cz - c0[2]])
      }
  return out
}

// Dirichlet 3x-domain clipping planes for the isosurface materials:
// keep A·x <= 3b ⇔ (−A_i/|A_i|)·x + 3b_i/|A_i| >= 0 (three clips distance < 0)
function useDomainClipPlanes(results: ComputeResponse, enabled: boolean): THREE.Plane[] {
  return useMemo(
    () =>
      enabled
        ? results.domainA.map((row, i) => {
            const n = Math.hypot(row[0], row[1], row[2])
            return new THREE.Plane(
              new THREE.Vector3(-row[0] / n, -row[1] / n, -row[2] / n),
              (3 * results.domainB[i]) / n,
            )
          })
        : [],
    [results, enabled],
  )
}

type IsoGeos = { included: THREE.BufferGeometry; ghost: THREE.BufferGeometry | null } | null

// Included/ghost geometries from an isosurface mesh: position/normal
// attributes are shared, so a filter change swaps only the index buffers.
// `filtered` ids live in the same space as roots/triOwner.
function makeIsoGeos(
  mesh: IsosurfaceMesh | null,
  roots: Int32Array | null,
  filtered: Set<number> | null,
): IsoGeos {
  if (!mesh || mesh.triangleCount === 0) return null
  const pos = new THREE.BufferAttribute(mesh.positions, 3)
  const nrm = new THREE.BufferAttribute(mesh.normals, 3)
  const included = new THREE.BufferGeometry()
  included.setAttribute('position', pos)
  included.setAttribute('normal', nrm)
  let ghost: THREE.BufferGeometry | null = null
  if (filtered && roots) {
    const split = splitTriangles(mesh, roots, filtered)
    if (split.includedIndex) included.setIndex(new THREE.BufferAttribute(split.includedIndex, 1))
    if (split.ghostIndex.length > 0) {
      ghost = new THREE.BufferGeometry()
      ghost.setAttribute('position', pos)
      ghost.setAttribute('normal', nrm)
      ghost.setIndex(new THREE.BufferAttribute(split.ghostIndex, 1))
    }
  }
  return { included, ghost }
}

// A translucent 3D family rendered with the depth pre-pass scheme (see
// ORDER_*/BIT_* below): which stencil bits and render orders it owns, and
// the opacity at which it shows through the other family.
interface PrepassFamily {
  frontBit: number
  ghostBit: number
  preOrder: number
  colorOrder: number
  ghostOrder: number
  ghostOpacity: number
}

// Presentational half of the isosurfaces: the surface instanced over the
// lattice translates covering the 3x domain and clipped to it, plus the
// world-space caps that close the solid where the domain boundary cuts it
// (slightly darkened, the classic cut-surface look). Parts dimmed by a
// subtree filter are drawn as plain low-opacity overlays. With `prepass`
// the family goes through the depth pre-pass passes — only its outer
// surface is shaded, occluding and showing through the other family
// correctly; without it, a plain glossy translucent material. Geometry
// DISPOSAL stays with the wrapper that created it.
function IsosurfaceMeshes({
  geos,
  capGeos,
  translates,
  planes,
  color,
  opacity,
  prepass,
}: {
  geos: IsoGeos
  capGeos: IsoGeos
  translates: [number, number, number][]
  planes: THREE.Plane[]
  color: string
  opacity: number
  prepass?: PrepassFamily
}) {
  // every instanced mesh of the surface (the passes and the dimmed part)
  const instRefs = useRef<(THREE.InstancedMesh | null)[]>([])
  const capColor = useMemo(
    () => '#' + new THREE.Color(color).multiplyScalar(0.72).getHexString(),
    [color],
  )
  const bits = prepass ?? { frontBit: 0, ghostBit: 0, ghostOpacity: 0 }
  const surfaceMats = useDepthPrepassMaterials(
    color,
    opacity,
    bits.frontBit,
    bits.ghostBit,
    bits.ghostOpacity,
    THREE.DoubleSide,
    planes,
  )
  const capMats = useDepthPrepassMaterials(
    capColor,
    opacity,
    bits.frontBit,
    bits.ghostBit,
    bits.ghostOpacity,
    THREE.DoubleSide,
  )

  // instance matrices are pure translations, independent of the threshold;
  // refilled when the translate list or a mesh (re)mounts
  const nT = translates.length
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    const m = new THREE.Matrix4()
    for (const inst of instRefs.current) {
      if (!inst) continue
      for (let i = 0; i < nT; i++) {
        m.makeTranslation(translates[i][0], translates[i][1], translates[i][2])
        inst.setMatrixAt(i, m)
      }
      inst.instanceMatrix.needsUpdate = true
    }
    // a frame may already have been drawn with the fresh mesh's unset matrices
    invalidate()
  }, [translates, nT, geos, prepass, invalidate])

  if ((!geos && !capGeos) || nT === 0) return null
  const dimOpacity = Math.min(0.1, opacity * 0.25)
  const inst = (
    k: number,
    geometry: THREE.BufferGeometry,
    material: THREE.Material | undefined,
    order: number,
    children?: React.ReactNode,
  ) => (
    <instancedMesh
      key={`iso-${k}-${nT}`}
      ref={(m) => {
        instRefs.current[k] = m
      }}
      args={[undefined, undefined, nT]}
      geometry={geometry}
      {...(material ? { material } : {})}
      frustumCulled={false}
      renderOrder={order}
    >
      {children}
    </instancedMesh>
  )
  const showGhost = prepass !== undefined && prepass.ghostOpacity > 0.003
  return (
    // without the pre-pass, renderOrder −1 draws before the other
    // transparents so ghost points and edge lines composite on top
    <>
      {geos &&
        (prepass ? (
          <>
            {inst(0, geos.included, surfaceMats.depthMat, prepass.preOrder)}
            {inst(1, geos.included, surfaceMats.colorMat, prepass.colorOrder)}
            {showGhost && inst(2, geos.included, surfaceMats.ghostMat, prepass.ghostOrder)}
          </>
        ) : (
          inst(
            1,
            geos.included,
            undefined,
            -1,
            <meshPhongMaterial
              color={color}
              specular="#777777"
              shininess={100}
              side={THREE.DoubleSide}
              transparent
              opacity={opacity}
              depthWrite={opacity > 0.99}
              clippingPlanes={planes}
            />,
          )
        ))}
      {geos?.ghost &&
        inst(
          3,
          geos.ghost,
          undefined,
          -1,
          <meshPhongMaterial
            color={color}
            specular="#777777"
            shininess={100}
            side={THREE.DoubleSide}
            transparent
            opacity={dimOpacity}
            depthWrite={false}
            clippingPlanes={planes}
          />,
        )}
      {capGeos &&
        (prepass ? (
          <>
            <mesh geometry={capGeos.included} material={capMats.depthMat} frustumCulled={false} renderOrder={prepass.preOrder} />
            <mesh geometry={capGeos.included} material={capMats.colorMat} frustumCulled={false} renderOrder={prepass.colorOrder} />
            {showGhost && (
              <mesh geometry={capGeos.included} material={capMats.ghostMat} frustumCulled={false} renderOrder={prepass.ghostOrder} />
            )}
          </>
        ) : (
          <mesh geometry={capGeos.included} frustumCulled={false} renderOrder={-1}>
            <meshPhongMaterial
              color={capColor}
              specular="#555555"
              shininess={60}
              side={THREE.DoubleSide}
              transparent
              opacity={opacity}
              depthWrite={opacity > 0.99}
            />
          </mesh>
        ))}
      {capGeos?.ghost && (
        <mesh geometry={capGeos.ghost} frustumCulled={false} renderOrder={-1}>
          <meshPhongMaterial
            color={capColor}
            specular="#555555"
            shininess={60}
            side={THREE.DoubleSide}
            transparent
            opacity={dimOpacity}
            depthWrite={false}
          />
        </mesh>
      )}
    </>
  )
}

// Sublevel-set isosurface of the grid field at the slider threshold (3D
// only): marching cubes over the periodic grid in index space, mapped by U,
// rendered as a glossy surface instanced over the lattice translates that
// cover the 3x domain and clipped to it by the Dirichlet halfspaces. Under a
// subtree filter, patches bounding components without a filtered vertex are
// ghosted (the labeling uses the same arc adjacency as the merge tree).
// With negate, the superlevel counterpart: the isosurface of the NEGATED
// field at the f_Sup threshold, on the superlevel subtree channel.
function GridIsosurface({
  results,
  radius,
  negate = false,
}: {
  results: ComputeResponse
  radius: number
  negate?: boolean
}) {
  const g = results.grid
  const is3d = results.d === 3
  const verts = useSubtreeVerts(negate ? 'voronoi' : 'delaunay')
  const opacity = useStore((s) => (negate ? s.ui.isoOpacitySup : s.ui.isoOpacity))
  // the lattice that produced these results (grid geometry is built from U,
  // not the reduced basis; inputs.lattice may have been edited since)
  const U = useStore((s) => s.computedLattice)

  const vals = useMemo(
    () => (g && negate ? g.values.map((v) => -v) : (g?.values ?? null)),
    [g, negate],
  )
  const field = useMemo(
    () => (g && vals && U && is3d ? buildGridField(g.shape, vals, U) : null),
    [g, vals, U, is3d],
  )
  const sortedArcs = useMemo(() => {
    if (!g || !is3d) return []
    // negate BEFORE the lower-star max: max(-f_u, -f_v) = -min(f_u, f_v)
    const arcs = negate
      ? g.arcs.map((a) => ({
          ...a,
          filtration: Math.max(-g.values[a.vStart], -g.values[a.vEnd]),
        }))
      : [...g.arcs]
    return arcs.sort((a, b) => a.filtration - b.filtration)
  }, [g, is3d, negate])
  const translates = useMemo(
    () => (U && is3d ? latticeTranslates3x(results, U) : []),
    [results, U, is3d],
  )
  const planes = useDomainClipPlanes(results, is3d)

  // radiusVor defaults to -Infinity ("at the slider minimum"): no surface
  const finite = Number.isFinite(radius)
  const t = radius + filtEps(radius)
  const mesh = useMemo(
    () => (field && finite ? marchingCubes(field, t) : null),
    [field, finite, t],
  )
  // component labeling is only needed while a subtree filter is active
  const roots = useMemo(
    () => (g && verts && finite ? labelSublevelComponents(sortedArcs, g.values.length, t) : null),
    [g, sortedArcs, t, verts, finite],
  )
  const geos = useMemo(() => makeIsoGeos(mesh, roots, verts), [mesh, roots, verts])
  // caps close the solid where the 3x domain boundary cuts the sublevel set
  const capMesh = useMemo(
    () =>
      field && U && finite
        ? buildDomainCaps(field, U, results.domainA, results.domainB, results.domain3x.vertices, t)
        : null,
    [field, U, finite, t, results],
  )
  const capGeos = useMemo(() => makeIsoGeos(capMesh, roots, verts), [capMesh, roots, verts])

  useEffect(() => {
    return () => {
      for (const gs of [geos, capGeos]) {
        gs?.included.dispose()
        gs?.ghost?.dispose()
      }
    }
  }, [geos, capGeos])

  if (!g || !is3d) return null
  return (
    <IsosurfaceMeshes
      geos={geos}
      capGeos={capGeos}
      translates={translates}
      planes={planes}
      color={negate ? ISO_COLOR_SUP : ISO_COLOR}
      opacity={opacity}
    />
  )
}

// Per-result geometry behind the point-set filtration surfaces, shared by
// the Delaunay and Voronoi channels: the power cells of the quotient sites,
// the 3x-domain facets cut by them, and the lattice translates to instance
// over. 3D point sets only.
interface PowerGeometry {
  cells: PowerCells
  caps: CapPieces
  translates: [number, number, number][]
  nVor: number // size of the Voronoi merge tree's vertex index space
}
const powerGeometryCache = new WeakMap<ComputeResponse, PowerGeometry>()

function powerGeometry(results: ComputeResponse): PowerGeometry {
  let g = powerGeometryCache.get(results)
  if (!g) {
    const { positions3x, kept, weights } = results.points
    // one position per quotient Voronoi vertex (any lattice copy will do)
    const vor: number[][] = []
    for (const a of results.voronoiGeometry?.arcs ?? []) {
      vor[a.vStart] = a.start
      vor[a.vEnd] = a.end
    }
    const cells = buildPowerCells({
      sites: kept.map((orig) => positions3x[orig]),
      weights: kept.map((orig) => weights[orig]),
      arcs: results.quotientArcs,
      basis: results.basis,
      vorVertices: results.voronoiGeometry ? vor : null,
    })
    const dv = results.domain3x.vertices
    g = {
      cells,
      caps: buildCapPieces(cells, results.domainA, results.domainB, dv),
      translates: surfaceTranslates(cells, results.domainA, results.domainB, dv),
      nVor: vor.length,
    }
    powerGeometryCache.set(results, g)
  }
  return g
}

// Filtration surfaces for 3D point sets, built exactly from the power cells
// (see powerSurface.ts): the level set of the power distance
// pi(x) = min_i(|x-p_i|^2 - w_i) is each site's sphere cut to its cell. At
// f_Del it bounds the Delaunay ball union; with negate, at f_Vor it bounds
// the Voronoi-filtration solid {pi >= -f_Vor}. Patches are linked to the
// merge trees through their quotient vertex (the site for the Delaunay
// channel, a Voronoi vertex of the bounded component for the Voronoi
// channel), with components labeled on the same arcs the trees are built on.
function PowerIsosurface({
  results,
  radius,
  negate = false,
}: {
  results: ComputeResponse
  radius: number
  negate?: boolean
}) {
  const is3d = results.d === 3
  const verts = useSubtreeVerts(negate ? 'voronoi' : 'delaunay')
  const opacity = useStore((s) => (negate ? s.ui.vorSurfaceOpacity : s.ui.delSurfaceOpacity))
  // the other family's opacity (0 when hidden): this one shows through it
  // at own·(1−other)
  const otherOpacity = useStore((s) =>
    negate
      ? s.ui.showDelSurface
        ? s.ui.delSurfaceOpacity
        : 0
      : s.ui.showVorSurface
        ? s.ui.vorSurfaceOpacity
        : 0,
  )
  const prepass = useMemo<PrepassFamily>(
    () =>
      negate
        ? {
            frontBit: BIT_VOR_FRONT,
            ghostBit: BIT_VOR_GHOST,
            preOrder: ORDER_VOR_PRE,
            colorOrder: ORDER_VOR_COLOR,
            ghostOrder: ORDER_VOR_GHOST,
            ghostOpacity: otherOpacity > 0 ? opacity * (1 - otherOpacity) : 0,
          }
        : {
            frontBit: BIT_DEL_FRONT,
            ghostBit: BIT_DEL_GHOST,
            preOrder: ORDER_DEL_PRE,
            colorOrder: ORDER_DEL_COLOR,
            ghostOrder: ORDER_DEL_GHOST,
            ghostOpacity: otherOpacity > 0 ? opacity * (1 - otherOpacity) : 0,
          },
    [negate, opacity, otherOpacity],
  )
  const geometry = useMemo(() => (is3d ? powerGeometry(results) : null), [results, is3d])
  const planes = useDomainClipPlanes(results, is3d)

  const sortedArcs = useMemo(() => {
    const arcs = negate ? (results.voronoiGeometry?.arcs ?? []) : results.quotientArcs
    return [...arcs].sort((a, b) => a.filtration - b.filtration)
  }, [results, negate])

  const finite = Number.isFinite(radius) // radiusVor defaults to -Infinity
  const t = radius + filtEps(radius)
  // both channels cut the same function: f_Vor lives on the -pi scale
  const level = negate ? -t : t
  const filtering = verts !== null
  const mesh = useMemo(
    () => (geometry && finite ? powerSurface(geometry.cells, level, negate, filtering) : null),
    [geometry, finite, level, negate, filtering],
  )
  // component labeling is only needed while a subtree filter is active
  const roots = useMemo(
    () =>
      geometry && verts && finite
        ? labelSublevelComponents(sortedArcs, negate ? geometry.nVor : geometry.cells.n, t)
        : null,
    [geometry, sortedArcs, negate, t, verts, finite],
  )
  const geos = useMemo(() => makeIsoGeos(mesh, roots, verts), [mesh, roots, verts])
  // caps close the solid where the 3x domain boundary cuts it
  const capMesh = useMemo(
    () =>
      geometry && finite
        ? powerCaps(geometry.cells, geometry.caps, level, negate, filtering)
        : null,
    [geometry, finite, level, negate, filtering],
  )
  const capGeos = useMemo(() => makeIsoGeos(capMesh, roots, verts), [capMesh, roots, verts])

  useEffect(() => {
    return () => {
      for (const gs of [geos, capGeos]) {
        gs?.included.dispose()
        gs?.ghost?.dispose()
      }
    }
  }, [geos, capGeos])

  if (!geometry) return null
  return (
    <IsosurfaceMeshes
      geos={geos}
      capGeos={capGeos}
      translates={geometry.translates}
      planes={planes}
      color={negate ? ISO_COLOR_SUP : ISO_COLOR}
      opacity={opacity}
      prepass={prepass}
    />
  )
}

// 2D drawing layers of the exact filtration regions (above the domain
// fill at -0.02, below the points and edges)
const REGION_Z_DEL = -0.0105
const REGION_Z_VOR = -0.0095

interface PowerGeometry2D {
  cells: PowerCells2D
  pieces: CapPieces
  nVor: number
}
const powerGeometry2DCache = new WeakMap<ComputeResponse, PowerGeometry2D>()

function powerGeometry2D(results: ComputeResponse): PowerGeometry2D {
  let g = powerGeometry2DCache.get(results)
  if (!g) {
    const { positions3x, kept, weights } = results.points
    const vor: number[][] = []
    for (const a of results.voronoiGeometry?.arcs ?? []) {
      vor[a.vStart] = a.start
      vor[a.vEnd] = a.end
    }
    const cells = buildPowerCells2D({
      sites: kept.map((orig) => positions3x[orig]),
      weights: kept.map((orig) => weights[orig]),
      arcs: results.quotientArcs,
      basis: results.basis,
      vorVertices: vor,
    })
    g = {
      cells,
      // in the z = 0 plane; each channel's mesh is lifted to its own layer
      pieces: buildRegionPieces2D(cells, (results.domain3x as Polytope2D).outline, 0),
      nVor: vor.length,
    }
    powerGeometry2DCache.set(results, g)
  }
  return g
}

// Exact filtration regions for 2D point sets, the flat counterpart of
// PowerIsosurface: the sublevel set {pi <= f_Del} of the power distance is
// the disk union, and with negate {pi >= -f_Vor} is the plane minus it.
// Both are drawn over the 3x domain as each power cell's part of the domain
// intersected with / minus its own disk (see powerSurface.ts). The pieces
// never overlap, so the flat translucent fill needs no stencil. Under a
// subtree filter, the parts of other components are ghosted, with
// components labeled on the arcs of the matching merge tree.
function PowerRegion2D({
  results,
  radius,
  negate = false,
}: {
  results: ComputeResponse
  radius: number
  negate?: boolean
}) {
  const verts = useSubtreeVerts(negate ? 'voronoi' : 'delaunay')
  const opacity = useStore((s) => (negate ? s.ui.vorSurfaceOpacity : s.ui.delSurfaceOpacity))
  const geometry = useMemo(() => powerGeometry2D(results), [results])
  const sortedArcs = useMemo(() => {
    const arcs = negate ? (results.voronoiGeometry?.arcs ?? []) : results.quotientArcs
    return [...arcs].sort((a, b) => a.filtration - b.filtration)
  }, [results, negate])

  const finite = Number.isFinite(radius) // radiusVor defaults to -Infinity
  const t = radius + filtEps(radius)
  // both channels cut the same function: f_Vor lives on the -pi scale
  const level = negate ? -t : t
  const filtering = verts !== null
  const mesh = useMemo(
    () =>
      finite ? powerRegion2D(geometry.cells, geometry.pieces, level, negate, filtering) : null,
    [geometry, finite, level, negate, filtering],
  )
  const roots = useMemo(
    () =>
      verts && finite
        ? labelSublevelComponents(sortedArcs, negate ? geometry.nVor : geometry.cells.n, t)
        : null,
    [geometry, sortedArcs, negate, t, verts, finite],
  )
  const geos = useMemo(() => makeIsoGeos(mesh, roots, verts), [mesh, roots, verts])
  useEffect(() => {
    return () => {
      geos?.included.dispose()
      geos?.ghost?.dispose()
    }
  }, [geos])

  if (!geos) return null
  const color = negate ? ISO_COLOR_SUP : ISO_COLOR
  const z = negate ? REGION_Z_VOR : REGION_Z_DEL
  return (
    <>
      <mesh geometry={geos.included} position={[0, 0, z]} frustumCulled={false}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={opacity}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      {geos.ghost && (
        <mesh geometry={geos.ghost} position={[0, 0, z]} frustumCulled={false}>
          <meshBasicMaterial
            color={color}
            transparent
            opacity={Math.min(0.1, opacity * 0.25)}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
    </>
  )
}

function VoronoiSkeleton({ results }: { results: ComputeResponse }) {
  const g = results.voronoiGeometry
  const z = results.d === 2 ? 0.003 : 0
  const geometry = useMemo(() => {
    if (!g) return null
    const pos: number[] = []
    for (const [s, t] of g.fullEdges) {
      const a = g.points3x[s]
      const b = g.points3x[t]
      pos.push(a[0], a[1], (a[2] ?? 0) + z, b[0], b[1], (b[2] ?? 0) + z)
    }
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    return geom
  }, [g, z])
  if (!geometry) return null
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={RED} transparent opacity={0.35} />
    </lineSegments>
  )
}

function VoronoiArcs({ results }: { results: ComputeResponse }) {
  const g = results.voronoiGeometry
  if (!g) return null
  const z = results.d === 2 ? 0.004 : 0
  return (
    <>
      {g.arcs.map((arc, i) => (
        <Line
          key={i}
          points={[
            [arc.start[0], arc.start[1], (arc.start[2] ?? 0) + z],
            [arc.end[0], arc.end[1], (arc.end[2] ?? 0) + z],
          ]}
          color={RED}
          lineWidth={2.5}
        />
      ))}
    </>
  )
}

// Voronoi points (circumcenters) across the 3x domain, light red —
// mirroring the light blue of the Delaunay point copies.
function VoronoiPoints({ results }: { results: ComputeResponse }) {
  const g = results.voronoiGeometry
  if (!g) return null
  return (
    <>
      {g.points3x.map((p, i) => (
        <mesh key={i} position={to3(p)}>
          <sphereGeometry args={[0.018, 16, 16]} />
          <meshBasicMaterial color="#e07f7f" />
        </mesh>
      ))}
    </>
  )
}

// Sublevel set of the Voronoi filtration at f_Vor (the Voronoi filtration
// lives on the negated power-distance scale, so thresholds are typically
// negative): the part of the Voronoi diagram not yet covered by the growing
// balls, tiled across the 3x domain.
function VoronoiFiltrationEdges({ results, radius }: { results: ComputeResponse; radius: number }) {
  const verts = useSubtreeVerts('voronoi')
  const data = useMemo(() => {
    const g = results.voronoiGeometry
    if (!g) return { points: [], filtration: [] } as TiledSegments
    const arcs = verts
      ? g.arcs.filter((a) => verts.has(a.vStart) && verts.has(a.vEnd))
      : g.arcs
    return buildTiledSegments(arcs, results, results.d === 2 ? 0.005 : 0)
  }, [results, verts])
  const opacity = useStore((s) => s.ui.vorEdgeOpacity)
  return <PrefixSegments data={data} threshold={radius} color={RED} opacity={opacity} />
}

// 3D pass ordering for the two translucent families (the Delaunay and
// Voronoi filtration surfaces). Both depth pre-passes run first, so the
// shared depth buffer holds the nearest surface of scene ∪ both solids;
// each color pass (EqualDepth) then shades exactly the pixels where its own
// family is that nearest surface — true mutual occlusion at full opacity.
// The ghost passes redraw each family where it lost the depth contest
// (GreaterDepth), at opacity own·(1−other), so a family shows through the
// other exactly to the extent the front one is transparent.
const ORDER_DEL_PRE = 1000
const ORDER_VOR_PRE = 1001
const ORDER_DEL_COLOR = 1002
const ORDER_VOR_COLOR = 1003
const ORDER_VOR_GHOST = 1004
const ORDER_DEL_GHOST = 1005
// 3D stencil bits: the color pass marks pixels where its family is the
// front surface; the ghost pass skips those pixels and marks its own so a
// solid's overlapping interior fragments blend only once.
const BIT_DEL_FRONT = 1
const BIT_VOR_FRONT = 2
const BIT_DEL_GHOST = 4
const BIT_VOR_GHOST = 8

// Depth pre-pass materials for a translucent 3D family. The depth material
// renders first (color writes off) and resolves the union's nearest surface
// per pixel in the shared depth buffer; the color material then shades
// exactly the pixels where this family won the depth contest (EqualDepth),
// so only the outer surface of the union is visible and each pixel is
// shaded once, and marks them in the stencil (frontBit). The ghost material
// redraws the family where it lost (GreaterDepth) at `ghostOpacity` =
// own·(1−other family's opacity); its stencil test skips pixels where this
// family is already the front surface and lets only the first fragment
// through (ghostBit via Invert), so the union's interior overlaps don't
// double-blend. All passes use the same material class so they rasterize
// bit-identical depths.
function useDepthPrepassMaterials(
  color: string,
  opacity: number,
  frontBit: number,
  ghostBit: number,
  ghostOpacity: number,
  side: THREE.Side = THREE.FrontSide,
  clippingPlanes: THREE.Plane[] = [],
) {
  const depthMat = useMemo(() => {
    const m = new THREE.MeshPhongMaterial({ transparent: true, depthWrite: true, side })
    m.colorWrite = false
    return m
  }, [side])
  useEffect(() => () => depthMat.dispose(), [depthMat])

  // soft shading: an emissive floor keeps faces pointing away from the
  // light close to the base color (the flat caps would otherwise go much
  // darker than the curved patches), and the low specular keeps the
  // highlights on the sphere patches from saturating
  const shadedPhong = () =>
    new THREE.MeshPhongMaterial({
      color,
      emissive: new THREE.Color(color).multiplyScalar(0.35),
      specular: '#454545',
      shininess: 32,
      transparent: true,
      depthWrite: false,
      side,
    })

  const colorMat = useMemo(() => {
    const m = shadedPhong()
    m.depthFunc = THREE.EqualDepth
    m.stencilWrite = true
    m.stencilFunc = THREE.AlwaysStencilFunc
    m.stencilRef = frontBit
    m.stencilWriteMask = frontBit
    m.stencilZPass = THREE.ReplaceStencilOp
    return m
  }, [color, frontBit, side])
  colorMat.opacity = opacity
  useEffect(() => () => colorMat.dispose(), [colorMat])

  const ghostMat = useMemo(() => {
    const m = shadedPhong()
    m.depthFunc = THREE.GreaterDepth
    m.stencilWrite = true
    m.stencilFunc = THREE.EqualStencilFunc
    m.stencilRef = 0
    m.stencilFuncMask = frontBit | ghostBit
    m.stencilWriteMask = ghostBit
    m.stencilZPass = THREE.InvertStencilOp
    return m
  }, [color, frontBit, ghostBit, side])
  ghostMat.opacity = ghostOpacity
  useEffect(() => () => ghostMat.dispose(), [ghostMat])

  // all passes must clip identically, or the depth contest is decided by
  // fragments the color pass never shades
  depthMat.clippingPlanes = clippingPlanes
  colorMat.clippingPlanes = clippingPlanes
  ghostMat.clippingPlanes = clippingPlanes

  return { depthMat, colorMat, ghostMat }
}

export default function Scene() {
  const results = useStore((s) => s.results)
  const ui = useStore((s) => s.ui)
  const status = useStore((s) => s.status)
  const hasGeometry = useStore((s) => s.hasGeometry)

  const extent = useMemo(() => {
    if (!results) return 2
    let m = 0
    for (const v of results.domain3x.vertices) for (const c of v) m = Math.max(m, Math.abs(c))
    return m || 2
  }, [results])

  if (!results) {
    const hint =
      status === 'loading'
        ? 'computing…'
        : hasGeometry
          ? 'press Compute to run the pipeline'
          : 'load a geometry file or generate a random input'
    return <div className="scene-placeholder">{hint}</div>
  }

  const is2d = results.d === 2

  return (
    <Canvas
      key={`${results.d}`}
      style={{ background: '#ffffff' }}
      gl={{ stencil: true }}
      frameloop="demand"
      onCreated={(state) => {
        captureRegistry.r3f = state
        // the grid isosurface is clipped to the 3x domain by material
        // clipping planes; set here so it survives the key remount on d change
        state.gl.localClippingEnabled = true
      }}
    >
      <InvalidateOnChange />
      {is2d ? (
        <>
          <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={220 / extent} />
          <MapControls enableRotate={false} screenSpacePanning />
        </>
      ) : (
        <>
          <OrbitControls makeDefault />
          <CameraSetup extent={extent} />
          <ambientLight intensity={1.0} />
          <directionalLight position={[3, 5, 4]} intensity={1.1} />
          {/* fill light opposite the key light, so surfaces facing away
              from it don't fall off to a much darker shade */}
          <directionalLight position={[-3, -2, -4]} intensity={0.45} />
        </>
      )}

      {ui.showBasis && <BasisArrows basis={results.basis} />}
      {ui.showDomains &&
        (is2d ? (
          <>
            <Domain2D polytope={results.domain1x as Polytope2D} fill z={-0.02} />
            <Domain2D polytope={results.domain3x as Polytope2D} fill={false} z={-0.02} />
          </>
        ) : (
          <>
            <Domain3D polytope={results.domain1x as Polytope3D} translucent />
            <Domain3D polytope={results.domain3x as Polytope3D} translucent={false} />
          </>
        ))}
      {results.grid ? (
        <>
          {ui.showFullSkeleton && <GridSkeleton results={results} />}
          {ui.showFiltrationEdges && <GridFiltrationEdges results={results} radius={ui.radius} />}
          {/* superlevel overlays (negated field, f_Sup on the -f scale);
              gated on the superlevel pass having succeeded */}
          {ui.showVoronoiFiltrationEdges && results.voronoi && (
            <GridFiltrationEdges results={results} radius={ui.radiusVor} negate />
          )}
          {ui.showPoints && <GridPoints results={results} radius={ui.radius} />}
          {!is2d && ui.showIsosurface && <GridIsosurface results={results} radius={ui.radius} />}
          {!is2d && ui.showIsosurfaceSup && results.voronoi && (
            <GridIsosurface results={results} radius={ui.radiusVor} negate />
          )}
        </>
      ) : (
        <>
          {ui.showFullSkeleton && <FullSkeleton results={results} />}
          {ui.showArcs && <QuotientArcs results={results} />}
          {ui.showFiltrationEdges && <FiltrationEdges results={results} radius={ui.radius} />}
          {ui.showVoronoiFiltrationEdges && <VoronoiFiltrationEdges results={results} radius={ui.radiusVor} />}
          {ui.showVoronoiSkeleton && <VoronoiSkeleton results={results} />}
          {ui.showVoronoiArcs && <VoronoiArcs results={results} />}
          {ui.showPoints && <Points results={results} />}
          {ui.showVoronoiPoints && <VoronoiPoints results={results} />}
          {!is2d && ui.showDelSurface && <PowerIsosurface results={results} radius={ui.radius} />}
          {!is2d && ui.showVorSurface && results.voronoiGeometry && (
            <PowerIsosurface results={results} radius={ui.radiusVor} negate />
          )}
          {is2d && ui.showDelSurface && <PowerRegion2D results={results} radius={ui.radius} />}
          {is2d && ui.showVorSurface && results.voronoiGeometry && (
            <PowerRegion2D results={results} radius={ui.radiusVor} negate />
          )}
        </>
      )}
    </Canvas>
  )
}

// Pop-up display options, embedded in the Visualization panel header.
// Column 1: static structures; column 2: slider-linked filtration overlays,
// each with its own transparency control.
const DISPLAY_TOGGLES = [
  { key: 'showBasis', label: 'lattice vectors' },
  { key: 'showDomains', label: 'Dirichlet domains' },
  { key: 'showPoints', label: 'Delaunay points' },
  { key: 'showVoronoiPoints', label: 'Voronoi points' },
  { key: 'showFullSkeleton', label: 'full Delaunay skeleton' },
  { key: 'showVoronoiSkeleton', label: 'full Voronoi skeleton' },
  { key: 'showArcs', label: 'periodic Delaunay edges' },
  { key: 'showVoronoiArcs', label: 'periodic Voronoi edges' },
] as const

// Filtration overlays are listed per filtration: the edges of one complex
// next to its exact surface/region, then the other complex.

// 3D point sets: the exact filtration surfaces
const FILTRATION_TOGGLES_3D = [
  { key: 'showFiltrationEdges', label: 'Delaunay filtration (edges)', opacityKey: 'filtEdgeOpacity' },
  { key: 'showDelSurface', label: 'Delaunay filtration (surface)', opacityKey: 'delSurfaceOpacity' },
  { key: 'showVoronoiFiltrationEdges', label: 'Voronoi filtration (edges)', opacityKey: 'vorEdgeOpacity' },
  { key: 'showVorSurface', label: 'Voronoi filtration (surface)', opacityKey: 'vorSurfaceOpacity' },
] as const

// 2D point sets: the exact filtration regions
const FILTRATION_TOGGLES_2D = [
  { key: 'showFiltrationEdges', label: 'Delaunay filtration (edges)', opacityKey: 'filtEdgeOpacity' },
  { key: 'showDelSurface', label: 'Delaunay filtration (region)', opacityKey: 'delSurfaceOpacity' },
  { key: 'showVoronoiFiltrationEdges', label: 'Voronoi filtration (edges)', opacityKey: 'vorEdgeOpacity' },
  { key: 'showVorSurface', label: 'Voronoi filtration (region)', opacityKey: 'vorSurfaceOpacity' },
] as const

// grid mode renders only points/skeleton/sublevel edges (plus the shared
// basis and domains), so the popup offers exactly those, with grid wording
const GRID_DISPLAY_TOGGLES = [
  { key: 'showBasis', label: 'lattice vectors' },
  { key: 'showDomains', label: 'Dirichlet domains' },
  { key: 'showPoints', label: 'grid points' },
  { key: 'showFullSkeleton', label: 'grid skeleton' },
] as const

const GRID_FILTRATION_TOGGLES = [
  { key: 'showFiltrationEdges', label: 'sublevel edges', opacityKey: 'filtEdgeOpacity' },
  { key: 'showVoronoiFiltrationEdges', label: 'superlevel edges', opacityKey: 'vorEdgeOpacity' },
] as const

// the isosurfaces exist only in 3D grid mode
const GRID_FILTRATION_TOGGLES_3D = [
  { key: 'showFiltrationEdges', label: 'sublevel edges', opacityKey: 'filtEdgeOpacity' },
  { key: 'showIsosurface', label: 'sublevel isosurface', opacityKey: 'isoOpacity' },
  { key: 'showVoronoiFiltrationEdges', label: 'superlevel edges', opacityKey: 'vorEdgeOpacity' },
  { key: 'showIsosurfaceSup', label: 'superlevel isosurface', opacityKey: 'isoOpacitySup' },
] as const

export function DisplayOptions() {
  const ui = useStore((s) => s.ui)
  const setUi = useStore((s) => s.setUi)
  const isGrid = useStore((s) => Boolean(s.results?.grid))
  const is3d = useStore((s) => s.results?.d === 3)
  const [open, setOpen] = useState(false)
  const displayToggles = isGrid ? GRID_DISPLAY_TOGGLES : DISPLAY_TOGGLES
  const filtrationToggles = isGrid
    ? is3d
      ? GRID_FILTRATION_TOGGLES_3D
      : GRID_FILTRATION_TOGGLES
    : is3d
      ? FILTRATION_TOGGLES_3D
      : FILTRATION_TOGGLES_2D
  return (
    <div className="popup-control">
      <button className={open ? 'active' : ''} onClick={() => setOpen(!open)}>
        display {open ? '▴' : '▾'}
      </button>
      {open && (
        <div className="popup-panel popup-columns">
          <div className="popup-col">
            {displayToggles.map(({ key, label }) => (
              <label key={key} className="row">
                <input type="checkbox" checked={ui[key]} onChange={(e) => setUi({ [key]: e.target.checked })} />
                {label}
              </label>
            ))}
          </div>
          <div className="popup-col">
            {filtrationToggles.map(({ key, label, opacityKey }) => (
              <div key={key}>
                <label className="row">
                  <input type="checkbox" checked={ui[key]} onChange={(e) => setUi({ [key]: e.target.checked })} />
                  {label}
                </label>
                {ui[key] && (
                  <div className="row popup-sub">
                    <span>transparency</span>
                    <input
                      type="range"
                      min={0.05}
                      max={1}
                      step={0.01}
                      value={ui[opacityKey]}
                      onChange={(e) => setUi({ [opacityKey]: e.target.valueAsNumber })}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// frameloop="demand": the scene repaints only when invalidated. The r3f
// reconciler invalidates on declarative prop changes and the drei controls
// invalidate on camera interaction, but several components mutate three.js
// state imperatively (material opacity, instanceCount via useFrame). Any
// store change that can affect the scene requests one frame here.
function InvalidateOnChange() {
  const invalidate = useThree((s) => s.invalidate)
  const gl = useThree((s) => s.gl)
  const ui = useStore((s) => s.ui)
  const results = useStore((s) => s.results)
  useEffect(() => {
    invalidate()
  }, [ui, results, invalidate])
  // debugging/testing probe (frame counter lives in gl.info.render.frame)
  useEffect(() => {
    ;(window as unknown as { __glInfo: typeof gl.info }).__glInfo = gl.info
  }, [gl])
  return null
}

function CameraSetup({ extent }: { extent: number }) {
  const camera = useThree((s) => s.camera)
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    camera.position.set(extent * 1.5, extent * 1.2, extent * 1.8)
    camera.lookAt(0, 0, 0)
    invalidate()
  }, [camera, extent, invalidate])
  return null
}
