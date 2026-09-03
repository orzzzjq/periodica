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
function PrefixSegments({ data, threshold, color }: { data: TiledSegments; threshold: number; color: string }) {
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
  return <Line ref={ref} points={data.points} segments color={color} lineWidth={5} visible={count > 0} />
}

// Sublevel set of the Delaunay filtration: the periodic edges whose
// power-scale filtration value is below the current threshold f_Del
// (slider-linked), tiled across the 3x Dirichlet domain.
function FiltrationEdges({ results, radius }: { results: ComputeResponse; radius: number }) {
  const data = useMemo(
    () => buildTiledSegments(results.quotientArcs, results, results.d === 2 ? 0.002 : 0),
    [results],
  )
  return <PrefixSegments data={data} threshold={radius} color={BLUE} />
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

// 2D cone approximation of the Voronoi filtration: every tiled Voronoi edge
// grows an isosceles triangle from each endpoint toward the other. A side
// starts once F = f_Vor passes the endpoint's vertex filtration f_V; its
// height along the edge is L·(F−f_V)/max(F−f_V, 2(f_E−f_V)) — reaching the
// midpoint exactly when the edge is born (F = f_E) and the far endpoint at
// the mirrored value 2f_E−f_V — and its base width is sqrt(F−f_V),
// centered at the Voronoi point.
interface ConeEdge {
  p1: [number, number]
  p2: [number, number]
  u: [number, number] // unit vector p1 -> p2
  n: [number, number] // unit normal
  L: number
  fE: number
  fV1: number
  fV2: number
}

const CONE_Z = -0.009 // same layer the red balls used in 2D

function VoronoiFiltrationCones({ results, radiusVor }: { results: ComputeResponse; radiusVor: number }) {
  const opacity = useStore((s) => s.ui.ballOpacity)

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
        const L = Math.hypot(dx, dy)
        if (L < 1e-12) continue
        const u: [number, number] = [dx / L, dy / L]
        out.push({
          p1: [a[0], a[1]],
          p2: [b[0], b[1]],
          u,
          n: [-u[1], u[0]],
          L,
          fE: arc.filtration,
          fV1: arc.fStart,
          fV2: arc.fEnd,
        })
      }
    }
    return out
  }, [results])

  // deduplicated tiled Voronoi points (a vertex is shared by several edge
  // copies; its disk depends only on F - f_V, so draw it once)
  const vertices = useMemo(() => {
    const m = new Map<string, { p: [number, number]; fV: number }>()
    for (const e of edges) {
      const ends = [
        { p: e.p1, fV: e.fV1 },
        { p: e.p2, fV: e.fV2 },
      ]
      for (const v of ends) {
        const key = `${v.p[0].toFixed(9)},${v.p[1].toFixed(9)}`
        if (!m.has(key)) m.set(key, v)
      }
    }
    return [...m.values()]
  }, [edges])

  const F = Number.isFinite(radiusVor) ? radiusVor : -Infinity

  const positions = useMemo(() => {
    const arr: number[] = []
    for (const e of edges) {
      const sides = [
        { c: e.p1, fV: e.fV1, dir: 1 },
        { c: e.p2, fV: e.fV2, dir: -1 },
      ]
      for (const { c, fV, dir } of sides) {
        const grow = F - fV
        if (grow <= 0) continue
        // linear growth at rate L / (2 (f_E - f_V)), clamped at L: reaches
        // the midpoint when the edge is born (F = f_E) and the full edge
        // length at the mirrored value F = 2 f_E - f_V
        const h = Math.max(0, Math.min(e.L, (grow * e.L) / (2 * (e.fE - fV))))
        const w2 = Math.sqrt(grow) / 2 // base width sqrt(F - f_V)
        arr.push(
          c[0] + e.n[0] * w2, c[1] + e.n[1] * w2, CONE_Z,
          c[0] - e.n[0] * w2, c[1] - e.n[1] * w2, CONE_Z,
          c[0] + dir * e.u[0] * h, c[1] + dir * e.u[1] * h, CONE_Z,
        )
      }
    }
    return new Float32Array(arr)
  }, [edges, F])

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    return g
  }, [positions])
  useEffect(() => () => geometry.dispose(), [geometry])

  // same flat-union stencil family as the red Voronoi balls (ref 2)
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

  // shared unit disk, scaled per vertex: diameter = the triangle base width
  const unitCircle = useMemo(() => new THREE.CircleGeometry(1, 48), [])
  useEffect(() => () => unitCircle.dispose(), [unitCircle])

  if (positions.length === 0) return null
  return (
    <group>
      <mesh geometry={geometry} material={material} />
      {vertices.map((v, i) => {
        const grow = F - v.fV
        if (grow <= 0) return null
        const r = Math.sqrt(grow) / 2 // radius = base width / 2
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

// Sublevel set of the Voronoi filtration at f_Vor (the Voronoi filtration
// lives on the negated power-distance scale, so thresholds are typically
// negative): the part of the Voronoi diagram not yet covered by the growing
// balls, tiled across the 3x domain.
function VoronoiFiltrationEdges({ results, radius }: { results: ComputeResponse; radius: number }) {
  const data = useMemo(() => {
    const g = results.voronoiGeometry
    if (!g) return { points: [], filtration: [] } as TiledSegments
    return buildTiledSegments(g.arcs, results, results.d === 2 ? 0.005 : 0)
  }, [results])
  return <PrefixSegments data={data} threshold={radius} color={RED} />
}

// Shared transparent-ball renderer.
// Stencil trick: each screen pixel is shaded by at most one ball of the same
// family (the first fragment marks the stencil with `stencilRef`, later
// fragments fail the NotEqual test), so overlapping balls render as a flat
// union instead of stacking alpha. Distinct families use distinct refs, so
// e.g. Delaunay (blue) and Voronoi (red) balls still blend with each other.
// In 3D the balls are lit (highlight + shading) so depth is readable.
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

  const material = useMemo(() => {
    const m = is2d
      ? new THREE.MeshBasicMaterial({ color, transparent: true, depthWrite: false })
      : new THREE.MeshPhongMaterial({
          color,
          specular: '#888888',
          shininess: 60,
          transparent: true,
          depthWrite: false,
        })
    m.stencilWrite = true
    m.stencilRef = stencilRef
    m.stencilFunc = THREE.NotEqualStencilFunc
    m.stencilZPass = THREE.ReplaceStencilOp
    return m
  }, [is2d, color, stencilRef])
  material.opacity = opacity
  useEffect(() => () => material.dispose(), [material])

  // The stencil makes the FIRST drawn fragment win, so draw balls nearest-
  // first: with shaded 3D spheres the closest surface must own the overlap.
  const groupRef = useRef<THREE.Group>(null)
  useFrame(({ camera }) => {
    if (is2d || !groupRef.current) return
    for (const m of groupRef.current.children) {
      m.renderOrder = m.position.distanceTo(camera.position)
    }
  })

  // One shared unit geometry, scaled per ball: rebuilding a SphereGeometry
  // for every ball on every slider tick is what made dragging stutter.
  const unitGeometry = useMemo(
    () => (is2d ? new THREE.CircleGeometry(1, 48) : new THREE.SphereGeometry(1, 32, 32)),
    [is2d],
  )
  useEffect(() => () => unitGeometry.dispose(), [unitGeometry])

  return (
    <group ref={groupRef}>
      {items.map(({ p, r }, i) => (
        <mesh
          key={i}
          position={is2d ? [p[0], p[1], z2d] : to3(p)}
          scale={r}
          material={material}
          geometry={unitGeometry}
        />
      ))}
    </group>
  )
}

// Delaunay filtration balls, light blue. The sublevel set of the power
// distance f_i(x) = ||x - p_i||^2 - w_i at f_Del is the union of balls of
// radius sqrt(f_Del + w_i); point i has no ball until f_Del >= -w_i.
function FiltrationBalls({ results, radius }: { results: ComputeResponse; radius: number }) {
  const { positions3x, originalIndex, weights, hidden } = results.points
  const hiddenSet = useMemo(() => new Set(hidden), [hidden])
  const items: { p: number[]; r: number }[] = []
  positions3x.forEach((p, i) => {
    const orig = originalIndex[i]
    if (hiddenSet.has(orig)) return
    const r2 = radius + weights[orig]
    if (r2 > 0) items.push({ p, r: Math.sqrt(r2) })
  })
  if (items.length === 0) return null
  return <Balls items={items} is2d={results.d === 2} color="#8fb0e8" stencilRef={1} z2d={-0.01} />
}

// Voronoi filtration balls: centered at the Voronoi points, uniform radius
// f_Vor - min(f_Vor), light red.
function VoronoiFiltrationBalls({ results, radiusVor }: { results: ComputeResponse; radiusVor: number }) {
  const g = results.voronoiGeometry
  let vMin = Infinity
  if (results.voronoi)
    for (const bars of results.voronoi.barcodes) for (const b of bars) vMin = Math.min(vMin, b.birth)
  if (!g || !Number.isFinite(vMin)) return null
  const r = (Number.isFinite(radiusVor) ? Math.max(radiusVor, vMin) : vMin) - vMin
  if (r <= 0) return null
  const items = g.points3x.map((p) => ({ p, r }))
  return <Balls items={items} is2d={results.d === 2} color="#e08f8f" stencilRef={2} z2d={-0.009} />
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
          <ambientLight intensity={1.2} />
          <directionalLight position={[3, 5, 4]} intensity={1.6} />
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
      {ui.showVoronoiBalls &&
        (is2d ? (
          <VoronoiFiltrationCones results={results} radiusVor={ui.radiusVor} />
        ) : (
          <VoronoiFiltrationBalls results={results} radiusVor={ui.radiusVor} />
        ))}
    </Canvas>
  )
}

// Pop-up display options, embedded in the Visualization panel header.
const DISPLAY_TOGGLES = [
  { key: 'showPoints', label: 'Delaunay points' },
  { key: 'showBasis', label: 'lattice vectors' },
  { key: 'showDomains', label: 'Dirichlet domains' },
  { key: 'showFullSkeleton', label: 'full Delaunay skeleton' },
  { key: 'showArcs', label: 'periodic Delaunay edges' },
  { key: 'showVoronoiPoints', label: 'Voronoi points' },
  { key: 'showVoronoiSkeleton', label: 'full Voronoi skeleton' },
  { key: 'showVoronoiArcs', label: 'periodic Voronoi edges' },
  { key: 'showBalls', label: 'Delaunay filtration (balls)' },
  { key: 'showFiltrationEdges', label: 'Delaunay filtration (edges)' },
  { key: 'showVoronoiBalls', label: 'Voronoi filtration (balls)' },
  { key: 'showVoronoiFiltrationEdges', label: 'Voronoi filtration (edges)' },
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
        <div className="popup-panel">
          {DISPLAY_TOGGLES.map(({ key, label }) => (
            <div key={key}>
              <label className="row">
                <input type="checkbox" checked={ui[key]} onChange={(e) => setUi({ [key]: e.target.checked })} />
                {label}
              </label>
              {key === 'showBalls' && ui.showBalls && (
                <div className="row popup-sub">
                  <span>transparency</span>
                  <input
                    type="range"
                    min={0.05}
                    max={1}
                    step={0.01}
                    value={ui.ballOpacity}
                    onChange={(e) => setUi({ ballOpacity: e.target.valueAsNumber })}
                  />
                </div>
              )}
            </div>
          ))}
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
