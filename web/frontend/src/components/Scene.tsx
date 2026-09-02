import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, MapControls, OrbitControls, OrthographicCamera } from '@react-three/drei'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
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
        const color = isHidden ? '#bbbbbb' : isCanonical ? '#111111' : '#888888'
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

// Sublevel set of the Delaunay filtration: the periodic edges whose
// filtration value is below the current ball radius R (slider-linked),
// tiled across the 3x Dirichlet domain.
function FiltrationEdges({ results, radius }: { results: ComputeResponse; radius: number }) {
  const segments = useMemo(() => {
    const arcs = results.quotientArcs.filter((a) => a.filtration <= radius + filtEps(radius))
    return tile3xSegments(arcs, results, results.d === 2 ? 0.002 : 0)
  }, [results, radius])

  if (segments.length === 0) return null
  // one LineSegments2 for all copies: per-copy <Line> components would
  // rebuild hundreds of geometries on every slider tick
  return <Line points={segments} segments color={BLUE} lineWidth={5} />
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

// The Voronoi filtration runs on the negated radius scale (cursor at
// x = -R): edges with filtration <= -R are the part of the Voronoi
// diagram not yet covered by the growing balls, tiled across the 3x domain.
function VoronoiFiltrationEdges({ results, radius }: { results: ComputeResponse; radius: number }) {
  const segments = useMemo(() => {
    const g = results.voronoiGeometry
    if (!g) return []
    const arcs = g.arcs.filter((a) => a.filtration <= -radius + filtEps(radius))
    return tile3xSegments(arcs, results, results.d === 2 ? 0.005 : 0)
  }, [results, radius])

  if (segments.length === 0) return null
  return <Line points={segments} segments color={RED} lineWidth={5} />
}

function FiltrationBalls({ results, radius }: { results: ComputeResponse; radius: number }) {
  const { positions3x, originalIndex, weights, hidden } = results.points
  const hiddenSet = useMemo(() => new Set(hidden), [hidden])
  const opacity = useStore((s) => s.ui.ballOpacity)
  const is2d = results.d === 2

  // Stencil trick: each screen pixel is shaded by at most one ball (the first
  // fragment marks the stencil, later ball fragments fail the NotEqual test),
  // so overlapping balls render as a flat union instead of stacking alpha.
  // In 3D the balls are lit (highlight + shading) so depth is readable.
  const material = useMemo(() => {
    const m = is2d
      ? new THREE.MeshBasicMaterial({ color: '#999999', transparent: true, depthWrite: false })
      : new THREE.MeshPhongMaterial({
          color: '#8899aa',
          specular: '#888888',
          shininess: 60,
          transparent: true,
          depthWrite: false,
        })
    m.stencilWrite = true
    m.stencilRef = 1
    m.stencilFunc = THREE.NotEqualStencilFunc
    m.stencilZPass = THREE.ReplaceStencilOp
    return m
  }, [is2d])
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

  if (radius <= 0 && weights.every((w) => w === 0)) return null
  return (
    <group ref={groupRef}>
      {positions3x.map((p, i) => {
        const orig = originalIndex[i]
        if (hiddenSet.has(orig)) return null
        const r = radius + Math.sqrt(weights[orig])
        if (r <= 0) return null
        return (
          <mesh
            key={i}
            position={is2d ? [p[0], p[1], -0.01] : to3(p)}
            scale={r}
            material={material}
            geometry={unitGeometry}
          />
        )
      })}
    </group>
  )
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
      {ui.showVoronoiFiltrationEdges && <VoronoiFiltrationEdges results={results} radius={ui.radius} />}
      {ui.showVoronoiSkeleton && <VoronoiSkeleton results={results} />}
      {ui.showVoronoiArcs && <VoronoiArcs results={results} />}
      {ui.showPoints && <Points results={results} />}
      {ui.showBalls && <FiltrationBalls results={results} radius={ui.radius} />}
    </Canvas>
  )
}

// Pop-up display options, embedded in the Visualization panel header.
const DISPLAY_TOGGLES = [
  { key: 'showPoints', label: 'points' },
  { key: 'showBasis', label: 'lattice vectors' },
  { key: 'showDomains', label: 'Dirichlet domains' },
  { key: 'showFullSkeleton', label: 'full Delaunay skeleton' },
  { key: 'showArcs', label: 'periodic Delaunay edges' },
  { key: 'showVoronoiSkeleton', label: 'full Voronoi skeleton' },
  { key: 'showVoronoiArcs', label: 'periodic Voronoi edges' },
  { key: 'showBalls', label: 'Delaunay filtration (balls)' },
  { key: 'showFiltrationEdges', label: 'Delaunay filtration (edges)' },
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
