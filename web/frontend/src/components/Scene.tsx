import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, MapControls, OrbitControls, OrthographicCamera } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import type { Line2 } from 'three-stdlib'
import { inDirichletDomain, type ComputeResponse, type Polytope2D, type Polytope3D } from '../api'
import { useStore } from '../store'

const GREEN = '#30b830'
const BLUE = '#0000fe'
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

const RED = '#dd2222'

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

// Cone approximation of the Voronoi filtration: every tiled Voronoi edge
// grows a cone (2D: isosceles triangle) from each endpoint toward the
// other. A side starts once F = f_Vor passes the endpoint's vertex
// filtration f_V; its (virtual) height along the edge grows linearly at
// rate L/(2(f_E−f_V)) — reaching the midpoint exactly when the edge is
// born (F = f_E) and the far endpoint at the mirrored value 2f_E−f_V. The
// virtual apex keeps extending past the far endpoint so the cone keeps
// thickening, but the drawn solid is truncated at edge length L: once the
// virtual height exceeds L it renders as a frustum (2D: trapezoid).
// The base (diameter sqrt(F−f_V), capped so it never exceeds
// the height of any cone grown from the same point) is centered at the
// Voronoi point, which also carries a disk/ball of the same diameter.
// The cap lives on the quotient (canonical) vertex: heights are computed
// once per quotient arc (all periodic copies of an arc share L, f_E, f_V,
// hence heights), each quotient vertex's cap is the minimum over its full
// quotient star, and every periodic copy of the vertex uses that cap — so
// copies near the tiling boundary, whose tiled star is incomplete, still
// get the same base diameter as the canonical copy.
interface ConeEdge {
  p1: [number, number, number]
  p2: [number, number, number]
  u: [number, number, number] // unit vector p1 -> p2
  n: [number, number] // in-plane unit normal (2D only)
  L: number
  fE: number
  fV1: number
  fV2: number
  qv1: number // quotient vertex index of p1
  qv2: number // quotient vertex index of p2
}

interface ConeSide {
  pos: [number, number, number] // base center = Voronoi vertex
  quat: THREE.Quaternion // rotates +Y onto the edge direction (3D only)
  fV: number
  denom: number // shared growth-rate denominator 2 f_E - (f_V1 + f_V2)
  L: number
  qv: number // quotient vertex index of the base center
}

const CONE_Z = -0.009 // 2D drawing layer (same the red balls used)

// virtual cone height — NOT clamped at the edge length L: past L the apex
// keeps extending (so the base-diameter cap keeps growing) and the drawn
// solid is truncated at L into a frustum.
// Both ends of an edge grow at the same rate L / (2 f_E - (f_V1 + f_V2)),
// so the two cones meet exactly at F = f_E, at the split point of the edge
// determined by the two endpoint filtrations:
//   h = (F - f_V) · L / denom,  denom = 2 f_E - (f_V1 + f_V2)
const coneHeight = (grow: number, L: number, denom: number) =>
  Math.max(0, (grow * L) / denom)

// shared growth-rate denominator of an edge's two cones
const coneDenom = (fE: number, fV1: number, fV2: number) => 2 * fE - (fV1 + fV2)

// top/bottom radius ratio of the frustum left when a virtual cone of
// height hVirt > L is cut at L (as hVirt → ∞ it approaches a cylinder)
const frustumRatio = (hVirt: number, L: number) =>
  Number.isFinite(hVirt) ? (hVirt - L) / hVirt : 1

function VoronoiFiltrationCones({ results, radiusVor }: { results: ComputeResponse; radiusVor: number }) {
  const opacity = useStore((s) => s.ui.coneOpacity)
  const is2d = results.d === 2
  // subtree view restriction: cones only along edges internal to the picked
  // component, disks/balls only on its vertices
  const verts = useSubtreeVerts('voronoi')
  const edgeIn = (qv1: number, qv2: number) => !verts || (verts.has(qv1) && verts.has(qv2))

  // static per-copy data: tiled once per compute result
  const edges = useMemo<ConeEdge[]>(() => {
    const g = results.voronoiGeometry
    if (!g) return []
    const out: ConeEdge[] = []
    for (const arc of g.arcs) {
      const segs = tile3xSegments([arc], results, 0) // pairs (start, end) per copy
      for (let k = 0; k < segs.length; k += 2) {
        const a = segs[k]
        const b = segs[k + 1]
        const dx = b[0] - a[0]
        const dy = b[1] - a[1]
        const dz = b[2] - a[2]
        const L = Math.hypot(dx, dy, dz)
        if (L < 1e-12) continue
        const u: [number, number, number] = [dx / L, dy / L, dz / L]
        out.push({
          p1: [a[0], a[1], a[2]],
          p2: [b[0], b[1], b[2]],
          u,
          n: [-u[1], u[0]],
          L,
          fE: arc.filtration,
          fV1: arc.fStart,
          fV2: arc.fEnd,
          qv1: arc.vStart,
          qv2: arc.vEnd,
        })
      }
    }
    return out
  }, [results])

  // deduplicated tiled Voronoi points (a vertex is shared by several edge
  // copies; its disk/ball depends only on F - f_V, so draw it once), each
  // tagged with its quotient vertex index.
  const vertices = useMemo(() => {
    const m = new Map<string, { p: [number, number, number]; fV: number; qv: number }>()
    for (const e of edges) {
      const ends = [
        { p: e.p1, fV: e.fV1, qv: e.qv1 },
        { p: e.p2, fV: e.fV2, qv: e.qv2 },
      ]
      for (const v of ends) {
        if (verts && !verts.has(v.qv)) continue
        const key = v.p.map((c) => c.toFixed(9)).join(',')
        if (!m.has(key)) m.set(key, v)
      }
    }
    return [...m.values()]
  }, [edges, verts])

  // 3D: one oriented cone per edge side, direction static, size slider-driven
  const coneSides = useMemo<ConeSide[]>(() => {
    if (is2d) return []
    const up = new THREE.Vector3(0, 1, 0)
    const out: ConeSide[] = []
    for (const e of edges) {
      if (!edgeIn(e.qv1, e.qv2)) continue
      const dir = new THREE.Vector3(...e.u)
      const denom = coneDenom(e.fE, e.fV1, e.fV2)
      out.push(
        {
          pos: e.p1,
          quat: new THREE.Quaternion().setFromUnitVectors(up, dir),
          fV: e.fV1,
          denom,
          L: e.L,
          qv: e.qv1,
        },
        {
          pos: e.p2,
          quat: new THREE.Quaternion().setFromUnitVectors(up, dir.clone().negate()),
          fV: e.fV2,
          denom,
          L: e.L,
          qv: e.qv2,
        },
      )
    }
    return out
  }, [edges, is2d, verts])

  const F = Number.isFinite(radiusVor) ? radiusVor : -Infinity

  // Per-quotient-vertex cap on the base diameter: heights of all cones are
  // computed first — once per quotient arc, since every periodic copy of an
  // arc shares L, f_E and f_V and hence heights — then each canonical
  // vertex's cap is the smallest current height over its full quotient star
  // (cones not yet born contribute height 0). All periodic copies of a
  // vertex share its cap.
  const quotientCap = useMemo(() => {
    const cap = new Map<number, number>()
    const g = results.voronoiGeometry
    if (!g) return cap
    for (const arc of g.arcs) {
      const L = Math.hypot(
        arc.end[0] - arc.start[0],
        arc.end[1] - arc.start[1],
        (arc.end[2] ?? 0) - (arc.start[2] ?? 0),
      )
      if (L < 1e-12) continue
      const sides = [
        { fV: arc.fStart, qv: arc.vStart },
        { fV: arc.fEnd, qv: arc.vEnd },
      ]
      const denom = coneDenom(arc.filtration, arc.fStart, arc.fEnd)
      for (const { fV, qv } of sides) {
        const grow = F - fV
        const h = grow <= 0 ? 0 : coneHeight(grow, L, denom)
        const prev = cap.get(qv)
        if (prev === undefined || h < prev) cap.set(qv, h)
      }
    }
    return cap
  }, [results, F])

  // base radius = half the capped diameter min(sqrt(F - f_V), cap)
  const baseRadius = (grow: number, qv: number) =>
    Math.min(Math.sqrt(grow), quotientCap.get(qv) ?? Infinity) / 2

  // 2D: all triangles in one buffer
  const positions = useMemo(() => {
    if (!is2d) return new Float32Array(0)
    const arr: number[] = []
    for (const e of edges) {
      if (!edgeIn(e.qv1, e.qv2)) continue
      const sides = [
        { c: e.p1, fV: e.fV1, dir: 1, qv: e.qv1 },
        { c: e.p2, fV: e.fV2, dir: -1, qv: e.qv2 },
      ]
      const denom = coneDenom(e.fE, e.fV1, e.fV2)
      for (const { c, fV, dir, qv } of sides) {
        const grow = F - fV
        if (grow <= 0) continue
        const hv = coneHeight(grow, e.L, denom) // virtual height
        const w2 = baseRadius(grow, qv) // half base width
        const b1x = c[0] + e.n[0] * w2
        const b1y = c[1] + e.n[1] * w2
        const b2x = c[0] - e.n[0] * w2
        const b2y = c[1] - e.n[1] * w2
        if (hv <= e.L) {
          // triangle: apex at the virtual height
          arr.push(
            b1x, b1y, CONE_Z,
            b2x, b2y, CONE_Z,
            c[0] + dir * e.u[0] * hv, c[1] + dir * e.u[1] * hv, CONE_Z,
          )
        } else {
          // trapezoid: cut at the far endpoint, top width by similarity
          const w2t = w2 * frustumRatio(hv, e.L)
          const tx = c[0] + dir * e.u[0] * e.L
          const ty = c[1] + dir * e.u[1] * e.L
          const t1x = tx + e.n[0] * w2t
          const t1y = ty + e.n[1] * w2t
          const t2x = tx - e.n[0] * w2t
          const t2y = ty - e.n[1] * w2t
          arr.push(
            b1x, b1y, CONE_Z, b2x, b2y, CONE_Z, t1x, t1y, CONE_Z,
            b2x, b2y, CONE_Z, t2x, t2y, CONE_Z, t1x, t1y, CONE_Z,
          )
        }
      }
    }
    return new Float32Array(arr)
  }, [edges, quotientCap, F, is2d, verts])

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    return g
  }, [positions])
  useEffect(() => () => geometry.dispose(), [geometry])

  // 2D: flat-union stencil family (ref 2)
  const material = useMemo(() => {
    const m = new THREE.MeshBasicMaterial({
      color: '#e08f8f',
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    m.stencilWrite = true
    m.stencilRef = 2
    m.stencilFunc = THREE.NotEqualStencilFunc
    m.stencilZPass = THREE.ReplaceStencilOp
    return m
  }, [])
  material.opacity = opacity
  useEffect(() => () => material.dispose(), [material])

  // 3D: depth pre-pass + ghost pass (see ORDER_*/BIT_*) — only the outer
  // surface of the cone+ball union is shaded; it occludes and is occluded
  // by the Delaunay ball family correctly, and where hidden behind it shows
  // through at a weight proportional to that family's transparency
  const ballOpacity = useStore((s) => s.ui.ballOpacity)
  const ballsShown = useStore((s) => s.ui.showBalls)
  const ghostOpacity = ballsShown ? opacity * (1 - ballOpacity) : 0
  const { depthMat, colorMat, ghostMat } = useDepthPrepassMaterials(
    '#e08f8f',
    opacity,
    BIT_VOR_FRONT,
    BIT_VOR_GHOST,
    ghostOpacity,
  )

  // shared unit shapes, scaled per use
  const unitCircle = useMemo(() => new THREE.CircleGeometry(1, 48), [])
  useEffect(() => () => unitCircle.dispose(), [unitCircle])
  const unitCone = useMemo(() => {
    // base center at the origin, apex at +Y = 1
    const g = new THREE.ConeGeometry(1, 1, 32)
    g.translate(0, 0.5, 0)
    return g
  }, [])
  useEffect(() => () => unitCone.dispose(), [unitCone])
  // unit frustums for cones cut at the edge length: base radius 1 at the
  // origin, top radius q at +Y = 1. q is quantized to 64 steps so slider
  // drags reuse cached geometries instead of rebuilding one per cone per
  // tick (that pattern is what made the ball slider stutter).
  const frustumCache = useMemo(() => new Map<number, THREE.CylinderGeometry>(), [])
  useEffect(
    () => () => {
      frustumCache.forEach((g) => g.dispose())
      frustumCache.clear()
    },
    [frustumCache],
  )
  const frustumGeometry = (q: number) => {
    const k = Math.max(0, Math.min(64, Math.round(q * 64)))
    let g = frustumCache.get(k)
    if (!g) {
      g = new THREE.CylinderGeometry(k / 64, 1, 1, 32)
      g.translate(0, 0.5, 0)
      frustumCache.set(k, g)
    }
    return g
  }
  const unitSphere = useMemo(() => new THREE.SphereGeometry(1, 32, 32), [])
  useEffect(() => () => unitSphere.dispose(), [unitSphere])

  if (is2d) {
    if (positions.length === 0) return null
    return (
      <group>
        <mesh geometry={geometry} material={material} />
        {vertices.map((v, i) => {
          const grow = F - v.fV
          if (grow <= 0) return null
          const r = baseRadius(grow, v.qv) // radius = capped base width / 2
          if (r <= 0) return null
          return (
            <mesh
              key={i}
              geometry={unitCircle}
              material={material}
              position={[v.p[0], v.p[1], CONE_Z]}
              scale={r}
            />
          )
        })}
      </group>
    )
  }

  // one full pass of the family (cones + vertex balls) per material
  const familyPass = (mat: THREE.Material, order: number, prefix: string) => (
    <>
      {coneSides.map((s, i) => {
        const grow = F - s.fV
        if (grow <= 0) return null
        const hv = coneHeight(grow, s.L, s.denom) // virtual height
        const r = baseRadius(grow, s.qv) // base radius = capped base width / 2
        if (r <= 0) return null
        const cut = hv > s.L
        return (
          <mesh
            key={`${prefix}c${i}`}
            geometry={cut ? frustumGeometry(frustumRatio(hv, s.L)) : unitCone}
            material={mat}
            position={s.pos}
            quaternion={s.quat}
            scale={[r, cut ? s.L : hv, r]}
            renderOrder={order}
          />
        )
      })}
      {vertices.map((v, i) => {
        const grow = F - v.fV
        if (grow <= 0) return null
        const r = baseRadius(grow, v.qv)
        if (r <= 0) return null
        return (
          <mesh
            key={`${prefix}s${i}`}
            geometry={unitSphere}
            material={mat}
            position={v.p}
            scale={r}
            renderOrder={order}
          />
        )
      })}
    </>
  )

  return (
    <group>
      {familyPass(depthMat, ORDER_VOR_PRE, 'd')}
      {familyPass(colorMat, ORDER_VOR_COLOR, 'k')}
      {ghostOpacity > 0.003 && familyPass(ghostMat, ORDER_VOR_GHOST, 'g')}
    </group>
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

// 3D pass ordering for the two translucent families (Delaunay balls,
// Voronoi cones+balls). Both depth pre-passes run first, so the shared
// depth buffer holds the nearest surface of scene ∪ both unions; each
// color pass (EqualDepth) then shades exactly the pixels where its own
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
// union's overlapping interior fragments blend only once (2D reuses the
// stencil buffer differently — flat refs 1/2 — the modes never coexist).
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
) {
  const depthMat = useMemo(() => {
    const m = new THREE.MeshPhongMaterial({ transparent: true, depthWrite: true })
    m.colorWrite = false
    return m
  }, [])
  useEffect(() => () => depthMat.dispose(), [depthMat])

  // soft shading: an emissive floor keeps faces pointing away from the
  // light close to the base color (cone flanks have near-constant normals
  // and would otherwise go much darker than the spheres), and the low
  // specular avoids the bright streak along a cone's slant
  const shadedPhong = () =>
    new THREE.MeshPhongMaterial({
      color,
      emissive: new THREE.Color(color).multiplyScalar(0.35),
      specular: '#454545',
      shininess: 32,
      transparent: true,
      depthWrite: false,
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
  }, [color, frontBit])
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
  }, [color, frontBit, ghostBit])
  ghostMat.opacity = ghostOpacity
  useEffect(() => () => ghostMat.dispose(), [ghostMat])

  return { depthMat, colorMat, ghostMat }
}

// Delaunay-ball renderer.
// 2D: stencil trick — each screen pixel is shaded by at most one ball of the
// family (the first fragment marks the stencil, later fragments fail the
// NotEqual test), a flat union. Distinct families use distinct refs and
// still blend with each other.
// 3D: depth pre-pass + ghost pass (see ORDER_*/BIT_* above) — the union's
// outer surface occludes and is occluded by the Voronoi family correctly,
// and shows through it to the extent that family is transparent.
function Balls({
  items,
  is2d,
  color,
  stencilRef,
  z2d,
}: {
  items: { p: number[]; r: number }[]
  is2d: boolean
  color: string
  stencilRef: number
  z2d: number
}) {
  const opacity = useStore((s) => s.ui.ballOpacity)
  const coneOpacity = useStore((s) => s.ui.coneOpacity)
  const conesShown = useStore((s) => s.ui.showVoronoiBalls)
  // weight of this family where it is hidden behind the Voronoi family
  const ghostOpacity = conesShown ? opacity * (1 - coneOpacity) : 0

  const flatMat = useMemo(() => {
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false })
    m.stencilWrite = true
    m.stencilRef = stencilRef
    m.stencilFunc = THREE.NotEqualStencilFunc
    m.stencilZPass = THREE.ReplaceStencilOp
    return m
  }, [color, stencilRef])
  flatMat.opacity = opacity
  useEffect(() => () => flatMat.dispose(), [flatMat])

  const { depthMat, colorMat, ghostMat } = useDepthPrepassMaterials(
    color,
    opacity,
    BIT_DEL_FRONT,
    BIT_DEL_GHOST,
    ghostOpacity,
  )

  // One shared unit geometry, scaled per ball: rebuilding a SphereGeometry
  // for every ball on every slider tick is what made dragging stutter.
  const unitGeometry = useMemo(
    () => (is2d ? new THREE.CircleGeometry(1, 48) : new THREE.SphereGeometry(1, 32, 32)),
    [is2d],
  )
  useEffect(() => () => unitGeometry.dispose(), [unitGeometry])

  if (is2d) {
    return (
      <group>
        {items.map(({ p, r }, i) => (
          <mesh key={i} position={[p[0], p[1], z2d]} scale={r} material={flatMat} geometry={unitGeometry} />
        ))}
      </group>
    )
  }
  const pass = (mat: THREE.Material, order: number, prefix: string) =>
    items.map(({ p, r }, i) => (
      <mesh
        key={`${prefix}${i}`}
        position={to3(p)}
        scale={r}
        material={mat}
        geometry={unitGeometry}
        renderOrder={order}
      />
    ))
  return (
    <group>
      {pass(depthMat, ORDER_DEL_PRE, 'd')}
      {pass(colorMat, ORDER_DEL_COLOR, 'c')}
      {ghostOpacity > 0.003 && pass(ghostMat, ORDER_DEL_GHOST, 'g')}
    </group>
  )
}

// Delaunay filtration balls, light blue. The sublevel set of the power
// distance f_i(x) = ||x - p_i||^2 - w_i at f_Del is the union of balls of
// radius sqrt(f_Del + w_i); point i has no ball until f_Del >= -w_i.
function FiltrationBalls({ results, radius }: { results: ComputeResponse; radius: number }) {
  const verts = useSubtreeVerts('delaunay')
  const { positions3x, originalIndex, weights, hidden, kept } = results.points
  const hiddenSet = useMemo(() => new Set(hidden), [hidden])
  // original point index -> quotient vertex index (the merge tree beam space)
  const origToQuotient = useMemo(() => new Map(kept.map((orig, qi) => [orig, qi])), [kept])
  const items: { p: number[]; r: number }[] = []
  positions3x.forEach((p, i) => {
    const orig = originalIndex[i]
    if (hiddenSet.has(orig)) return
    if (verts && !verts.has(origToQuotient.get(orig) ?? -1)) return
    const r2 = radius + weights[orig]
    if (r2 > 0) items.push({ p, r: Math.sqrt(r2) })
  })
  if (items.length === 0) return null
  return <Balls items={items} is2d={results.d === 2} color="#8fb0e8" stencilRef={1} z2d={-0.01} />
}

export default function Scene() {
  const results = useStore((s) => s.results)
  const ui = useStore((s) => s.ui)

  const extent = useMemo(() => {
    if (!results) return 2
    let m = 0
    for (const v of results.domain3x.vertices) for (const c of v) m = Math.max(m, Math.abs(c))
    return m || 2
  }, [results])

  if (!results) return <div className="scene-placeholder">computing…</div>

  const is2d = results.d === 2

  return (
    <Canvas key={`${results.d}`} style={{ background: '#ffffff' }} gl={{ stencil: true }}>
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
              from it (e.g. cone flanks) don't fall off to a much darker
              shade than the spheres */}
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
      {ui.showFullSkeleton && <FullSkeleton results={results} />}
      {ui.showArcs && <QuotientArcs results={results} />}
      {ui.showFiltrationEdges && <FiltrationEdges results={results} radius={ui.radius} />}
      {ui.showVoronoiFiltrationEdges && <VoronoiFiltrationEdges results={results} radius={ui.radiusVor} />}
      {ui.showVoronoiSkeleton && <VoronoiSkeleton results={results} />}
      {ui.showVoronoiArcs && <VoronoiArcs results={results} />}
      {ui.showPoints && <Points results={results} />}
      {ui.showVoronoiPoints && <VoronoiPoints results={results} />}
      {ui.showBalls && <FiltrationBalls results={results} radius={ui.radius} />}
      {ui.showVoronoiBalls && <VoronoiFiltrationCones results={results} radiusVor={ui.radiusVor} />}
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

const FILTRATION_TOGGLES = [
  { key: 'showBalls', label: 'Delaunay filtration (balls)', opacityKey: 'ballOpacity' },
  { key: 'showFiltrationEdges', label: 'Delaunay filtration (edges)', opacityKey: 'filtEdgeOpacity' },
  { key: 'showVoronoiBalls', label: 'Voronoi filtration (cones)', opacityKey: 'coneOpacity' },
  { key: 'showVoronoiFiltrationEdges', label: 'Voronoi filtration (edges)', opacityKey: 'vorEdgeOpacity' },
] as const

export function DisplayOptions() {
  const ui = useStore((s) => s.ui)
  const setUi = useStore((s) => s.setUi)
  const [open, setOpen] = useState(false)
  return (
    <div className="popup-control">
      <button className={open ? 'active' : ''} onClick={() => setOpen(!open)}>
        display {open ? '▴' : '▾'}
      </button>
      {open && (
        <div className="popup-panel popup-columns">
          <div className="popup-col">
            {DISPLAY_TOGGLES.map(({ key, label }) => (
              <label key={key} className="row">
                <input type="checkbox" checked={ui[key]} onChange={(e) => setUi({ [key]: e.target.checked })} />
                {label}
              </label>
            ))}
          </div>
          <div className="popup-col">
            {FILTRATION_TOGGLES.map(({ key, label, opacityKey }) => (
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

function CameraSetup({ extent }: { extent: number }) {
  const camera = useThree((s) => s.camera)
  useEffect(() => {
    camera.position.set(extent * 1.5, extent * 1.2, extent * 1.8)
    camera.lookAt(0, 0, 0)
  }, [camera, extent])
  return null
}
