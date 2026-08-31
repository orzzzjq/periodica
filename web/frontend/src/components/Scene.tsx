import { Canvas, useThree } from '@react-three/fiber'
import { Line, MapControls, OrbitControls, OrthographicCamera } from '@react-three/drei'
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import type { ComputeResponse, Polytope2D, Polytope3D } from '../api'
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
      <lineBasicMaterial color="#444444" transparent opacity={0.3} />
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

function FiltrationBalls({ results, radius }: { results: ComputeResponse; radius: number }) {
  const { positions3x, originalIndex, weights, hidden } = results.points
  const hiddenSet = useMemo(() => new Set(hidden), [hidden])
  const is2d = results.d === 2
  if (radius <= 0 && weights.every((w) => w === 0)) return null
  return (
    <>
      {positions3x.map((p, i) => {
        const orig = originalIndex[i]
        if (hiddenSet.has(orig)) return null
        const r = radius + Math.sqrt(weights[orig])
        if (r <= 0) return null
        return (
          <mesh key={i} position={is2d ? [p[0], p[1], -0.01] : to3(p)}>
            {is2d ? <circleGeometry args={[r, 48]} /> : <sphereGeometry args={[r, 24, 24]} />}
            <meshBasicMaterial color="#999999" transparent opacity={is2d ? 0.35 : 0.18} depthWrite={false} />
          </mesh>
        )
      })}
    </>
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
    <Canvas key={`${results.d}`} style={{ background: '#ffffff' }}>
      {is2d ? (
        <>
          <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={220 / extent} />
          <MapControls enableRotate={false} screenSpacePanning />
        </>
      ) : (
        <>
          <OrbitControls makeDefault />
          <CameraSetup extent={extent} />
        </>
      )}

      <BasisArrows basis={results.basis} />
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
      <QuotientArcs results={results} />
      <Points results={results} />
      {ui.showBalls && <FiltrationBalls results={results} radius={ui.radius} />}
    </Canvas>
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
