// Property checks for src/powerSurface.ts against a live backend: the exact
// filtration surfaces must lie on the level set of the power distance, cover
// all of it, cap the 3x domain consistently, and attribute every patch to
// the right merge-tree component.
// Usage: node scripts/power-surface-check.ts   (backend from PERIODICA_URL)
import {
  buildCapPieces,
  buildPowerCells,
  buildPowerCells2D,
  buildRegionPieces2D,
  powerCaps,
  powerRegion2D,
  powerSurface,
  surfaceTranslates,
} from '../src/powerSurface.ts'

const BASE = process.env.PERIODICA_URL ?? 'http://localhost:8000'

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

let failures = 0
function check(ok: boolean, what: string) {
  if (!ok) {
    failures++
    console.log(`  FAIL ${what}`)
  }
}

const LATTICES: Record<string, number[][]> = {
  cubic: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  skew: [[1, 0.2, 0.1], [0, 0.9, 0.3], [0, 0, 1.1]],
}

async function runCase(name: string, lattice: number[][], nPts: number, wMax: number, seed: number) {
  const rnd = mulberry32(seed)
  const frac = Array.from({ length: nPts }, () => [rnd(), rnd(), rnd()])
  const points = frac.map((f) => lattice.map((row) => row[0] * f[0] + row[1] * f[1] + row[2] * f[2]))
  const weights = frac.map(() => wMax * rnd())
  const res = await fetch(`${BASE}/api/compute`, {
    method: 'POST',
    // no keep-alive: the server drops idle connections while a case computes
    headers: { 'content-type': 'application/json', connection: 'close' },
    body: JSON.stringify({ d: 3, lattice, points, weights, imageSize: 20 }),
  })
  if (!res.ok) throw new Error(`${name}: backend ${res.status} ${await res.text()}`)
  const r = await res.json()

  const { positions3x, kept } = r.points
  const sites: number[][] = kept.map((o: number) => positions3x[o])
  const w: number[] = kept.map((o: number) => r.points.weights[o])
  const vor: number[][] = []
  const fVor: number[] = []
  for (const a of r.voronoiGeometry.arcs) {
    vor[a.vStart] = a.start
    vor[a.vEnd] = a.end
    fVor[a.vStart] = a.fStart
    fVor[a.vEnd] = a.fEnd
  }
  const t0 = performance.now()
  const cells = buildPowerCells({ sites, weights: w, arcs: r.quotientArcs, basis: r.basis, vorVertices: vor })
  const caps = buildCapPieces(cells, r.domainA, r.domainB, r.domain3x.vertices)
  const translates = surfaceTranslates(cells, r.domainA, r.domainB, r.domain3x.vertices)
  const tBuild = performance.now() - t0
  const n = cells.n
  const nDirs = cells.dirs.length / 3
  // the same cells on a sphere one subdivision finer, as an area reference
  const fineLevel = Math.round(Math.log(cells.tris.length / 60) / Math.log(4)) + 1
  const fine = buildPowerCells({
    sites, weights: w, arcs: r.quotientArcs, basis: r.basis, vorVertices: null, level: fineLevel,
  })
  console.log(
    `${name}: ${n} sites, ${vor.length} Voronoi vertices, sphere ${cells.tris.length / 3} tris, ` +
      `${caps.count} cap pieces, ${translates.length} translates, setup ${tBuild.toFixed(0)} ms`,
  )

  // brute-force power distance over the site copies within 3 lattice steps
  const B = r.basis as number[][]
  const shifts: number[][] = []
  for (let a = -3; a <= 3; a++)
    for (let b = -3; b <= 3; b++)
      for (let c = -3; c <= 3; c++)
        shifts.push([0, 1, 2].map((j) => a * B[0][j] + b * B[1][j] + c * B[2][j]))
  const zero = shifts.findIndex((s) => s[0] === 0 && s[1] === 0 && s[2] === 0)
  // returns [pi, site, shift index] of the minimizer
  const power = (x: number, y: number, z: number): [number, number, number] => {
    let best = Infinity
    let bi = -1
    let bs = -1
    for (let i = 0; i < n; i++) {
      for (let s = 0; s < shifts.length; s++) {
        const dx = x - sites[i][0] - shifts[s][0]
        const dy = y - sites[i][1] - shifts[s][1]
        const dz = z - sites[i][2] - shifts[s][2]
        const v = dx * dx + dy * dy + dz * dz - w[i]
        if (v < best) {
          best = v
          bi = i
          bs = s
        }
      }
    }
    return [best, bi, bs]
  }

  // 1. the planes describe the power cells: a point is inside cell i exactly
  //    when site i (unshifted) is its power-nearest site
  let cellMismatch = 0
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 400; k++) {
      const R = cells.radius[i] * 1.1
      const y = [R * (2 * rnd() - 1), R * (2 * rnd() - 1), R * (2 * rnd() - 1)]
      let margin = Infinity
      for (let p = cells.planeStart[i]; p < cells.planeStart[i + 1]; p++) {
        const nn = Math.hypot(cells.planes[4 * p], cells.planes[4 * p + 1], cells.planes[4 * p + 2])
        margin = Math.min(
          margin,
          (cells.planes[4 * p + 3] -
            (cells.planes[4 * p] * y[0] + cells.planes[4 * p + 1] * y[1] + cells.planes[4 * p + 2] * y[2])) /
            nn,
        )
      }
      if (Math.abs(margin) < 1e-6) continue // too close to a face to call
      const [, bi, bs] = power(sites[i][0] + y[0], sites[i][1] + y[1], sites[i][2] + y[2])
      if (margin > 0 !== (bi === i && bs === zero)) cellMismatch++
    }
  }
  check(cellMismatch === 0, `cells: ${cellMismatch} sample points disagree with the brute-force nearest site`)
  let unnamed = 0
  for (let v = 0; v < cells.vertVor.length; v++) if (cells.vertVor[v] < 0) unnamed++
  check(unnamed === 0, `${unnamed} of ${cells.vertVor.length} cell vertices not matched to a Voronoi vertex`)

  // largest angle between adjacent sphere directions (sets the chord error)
  let cosMin = 1
  for (let f = 0; f < cells.tris.length; f += 3)
    for (let e = 0; e < 3; e++) {
      const a = cells.tris[f + e]
      const b = cells.tris[f + ((e + 1) % 3)]
      cosMin = Math.min(
        cosMin,
        cells.dirs[3 * a] * cells.dirs[3 * b] +
          cells.dirs[3 * a + 1] * cells.dirs[3 * b + 1] +
          cells.dirs[3 * a + 2] * cells.dirs[3 * b + 2],
      )
    }
  const sin2Half = (1 - cosMin) / 2 // sin^2 of half the largest edge angle

  const triArea = (P: Float32Array, o: number) => {
    const ux = P[o + 3] - P[o]
    const uy = P[o + 4] - P[o + 1]
    const uz = P[o + 5] - P[o + 2]
    const vx = P[o + 6] - P[o]
    const vy = P[o + 7] - P[o + 1]
    const vz = P[o + 8] - P[o + 2]
    return 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
  }
  const pieceArea = (() => {
    let s = 0
    for (let p = 0; p < caps.count; p++) {
      for (let v = caps.polyStart[p]; v < caps.polyStart[p + 1]; v++) {
        const wv = v + 1 === caps.polyStart[p + 1] ? caps.polyStart[p] : v + 1
        s += 0.5 * (caps.poly[2 * v] * caps.poly[2 * wv + 1] - caps.poly[2 * wv] * caps.poly[2 * v + 1])
      }
    }
    return s
  })()
  let domainArea = 0
  for (const tri of r.domain3x.triangles) {
    const [a, b, c] = tri.map((i: number) => r.domain3x.vertices[i])
    domainArea += 0.5 * Math.hypot(
      (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
      (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
      (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
    )
  }
  check(
    Math.abs(pieceArea - domainArea) < 1e-6 * domainArea,
    `cap pieces tile the 3x boundary: ${pieceArea} vs ${domainArea}`,
  )

  // levels across the whole range of the power distance at the Voronoi vertices
  const piTop = Math.max(...fVor.map((f) => -f))
  const piBot = -Math.max(...w)
  const edgeF: number[] = r.voronoiGeometry.arcs.map((a: { filtration: number }) => a.filtration)
  const levels = [0.08, 0.3, 0.55, 0.8, 0.97].map((q) => piBot + q * (piTop - piBot))
  // ... plus just after a Voronoi edge is born, where the tubes are thinnest
  const fresh = edgeF.sort((a, b) => a - b)[Math.floor(edgeF.length / 3)]
  levels.push(-(fresh + 1e-4 * (piTop - piBot)))

  let tSurf = 0
  let tCaps = 0
  for (const level of levels) {
    let t1 = performance.now()
    const del = powerSurface(cells, level, false, false)
    tSurf += performance.now() - t1
    const tag = `level ${level.toFixed(4)}`

    // 2. every vertex is on the level set, in its own site's cell
    let worst = 0
    let rMax2 = 0
    for (let i = 0; i < n; i++) rMax2 = Math.max(rMax2, level + w[i])
    for (let t = 0; t < del.triangleCount; t++) {
      for (let c = 0; c < 3; c++) {
        const o = 9 * t + 3 * c
        const [pi] = power(del.positions[o], del.positions[o + 1], del.positions[o + 2])
        worst = Math.max(worst, Math.abs(pi - level))
      }
      if (t > 4000 && t % 7) continue // subsample large meshes
    }
    // corners where two faces meet inside a (refined) sphere triangle are
    // placed along a great arc instead of the crease circle: second order
    // in the refined triangle size
    check(worst <= 4 * rMax2 * (cells.spread / 16) + 1e-6, `${tag}: vertex off the level set by ${worst}`)

    // 3. the patches cover the whole level set: area vs direction sampling
    let area = 0
    for (let t = 0; t < del.triangleCount; t++) area += triArea(del.positions, 9 * t)
    let exact = 0
    let insideTotal = 0
    const S = 20000
    for (let i = 0; i < n; i++) {
      const r2 = level + w[i]
      if (!(r2 > 0)) continue
      const rad = Math.sqrt(r2)
      let inside = 0
      for (let k = 0; k < S; k++) {
        const z = 2 * rnd() - 1
        const ph = 2 * Math.PI * rnd()
        const s = Math.sqrt(1 - z * z)
        const y = [rad * s * Math.cos(ph), rad * s * Math.sin(ph), rad * z]
        let ok = true
        for (let p = cells.planeStart[i]; p < cells.planeStart[i + 1] && ok; p++)
          if (
            cells.planes[4 * p] * y[0] + cells.planes[4 * p + 1] * y[1] + cells.planes[4 * p + 2] * y[2] >
            cells.planes[4 * p + 3]
          )
            ok = false
        if (ok) inside++
      }
      insideTotal += inside
      exact += 4 * Math.PI * r2 * (inside / S)
    }
    // (the sampling is only trusted where enough directions land inside)
    const relArea = exact > 0 ? Math.abs(area - exact) / exact : area
    if (insideTotal >= 8000)
      check(relArea < 0.04, `${tag}: surface area ${area.toFixed(4)} vs sampled ${exact.toFixed(4)}`)
    // ... and against a twice finer sphere: flat facets lose O(θ^2) of the area
    const fineMesh = powerSurface(fine, level, false, false)
    let areaFine = 0
    for (let t = 0; t < fineMesh.triangleCount; t++) areaFine += triArea(fineMesh.positions, 9 * t)
    const relFine = areaFine > 0 ? Math.abs(area - areaFine) / areaFine : area
    // (patches smaller than a refined triangle are only resolved to its sagitta)
    if (areaFine > 1e-5 * rMax2)
      check(relFine < 4 * sin2Half + 1e-4, `${tag}: area ${area} vs ${areaFine} on the finer sphere`)

    // 4. the Voronoi channel is the same surface seen from the other side
    t1 = performance.now()
    const vorMesh = powerSurface(cells, level, true, true)
    tSurf += performance.now() - t1
    check(vorMesh.triangleCount === del.triangleCount, `${tag}: channel triangle counts differ`)

    // 5. owners: alive Voronoi vertices, one merge-tree component per patch
    const F = -level
    const parent = Int32Array.from({ length: vor.length }, (_, v) => v)
    const find = (v: number) => {
      while (parent[v] !== v) v = parent[v] = parent[parent[v]]
      return v
    }
    for (const a of r.voronoiGeometry.arcs) if (a.filtration <= F) parent[find(a.vStart)] = find(a.vEnd)
    let badOwner = 0
    const keyRoot = new Map<string, number>()
    let mixed = 0
    for (let t = 0; t < vorMesh.triangleCount; t++) {
      const own = vorMesh.triOwner[t]
      if (own >= vor.length || !(fVor[own] <= F + 1e-9)) {
        badOwner++
        continue
      }
      const root = find(own)
      // triangles sharing a corner belong to one connected patch
      for (let c = 0; c < 3; c++) {
        const o = 9 * t + 3 * c
        const key = `${vorMesh.positions[o].toFixed(5)},${vorMesh.positions[o + 1].toFixed(5)},${vorMesh.positions[o + 2].toFixed(5)}`
        const seen = keyRoot.get(key)
        if (seen === undefined) keyRoot.set(key, root)
        else if (seen !== root) mixed++
      }
    }
    check(badOwner === 0, `${tag}: ${badOwner} triangles owned by a missing or unborn Voronoi vertex`)
    check(mixed === 0, `${tag}: ${mixed} shared corners join patches of different components`)

    // 6. caps: on the 3x boundary, on the right side of the level, and the
    //    two channels split the boundary between them
    t1 = performance.now()
    const capDel = powerCaps(cells, caps, level, false, false)
    const capVor = powerCaps(cells, caps, level, true, true)
    tCaps += performance.now() - t1
    let wrongSide = 0
    let offBoundary = 0
    const sideTol = 1e-3 * (piTop - piBot)
    for (const [mesh, sign] of [[capDel, 1], [capVor, -1]] as const) {
      for (let t = 0; t < mesh.triangleCount; t++) {
        const o = 9 * t
        if (triArea(mesh.positions, o) < 1e-10) continue
        const x = (mesh.positions[o] + mesh.positions[o + 3] + mesh.positions[o + 6]) / 3
        const y = (mesh.positions[o + 1] + mesh.positions[o + 4] + mesh.positions[o + 7]) / 3
        const z = (mesh.positions[o + 2] + mesh.positions[o + 5] + mesh.positions[o + 8]) / 3
        if (sign * (power(x, y, z)[0] - level) > sideTol) {
          wrongSide++
          if (wrongSide <= 3)
            console.log(
              `    wrong side: ${sign > 0 ? 'del' : 'vor'} cap tri ${t} at (${x.toFixed(4)}, ${y.toFixed(4)}, ${z.toFixed(4)}) ` +
                `pi-level ${(power(x, y, z)[0] - level).toExponential(2)} area ${triArea(mesh.positions, o).toExponential(2)} owner ${mesh.triOwner[t]}`,
            )
        }
        let onFace = false
        let inDomain = true
        for (let h = 0; h < r.domainA.length; h++) {
          const a = r.domainA[h]
          const v = a[0] * x + a[1] * y + a[2] * z - 3 * r.domainB[h]
          const an = Math.hypot(a[0], a[1], a[2])
          if (Math.abs(v) < 1e-5 * an) onFace = true
          if (v > 1e-5 * an) inDomain = false
        }
        if (!onFace || !inDomain) offBoundary++
      }
    }
    check(wrongSide === 0, `${tag}: ${wrongSide} cap triangles on the wrong side of the level`)
    check(offBoundary === 0, `${tag}: ${offBoundary} cap triangles off the 3x boundary`)
    let aDel = 0
    for (let t = 0; t < capDel.triangleCount; t++) aDel += triArea(capDel.positions, 9 * t)
    let aVor = 0
    for (let t = 0; t < capVor.triangleCount; t++) aVor += triArea(capVor.positions, 9 * t)
    check(
      Math.abs(aDel + aVor - pieceArea) < 2e-3 * pieceArea,
      `${tag}: cap areas ${aDel.toFixed(4)} + ${aVor.toFixed(4)} != boundary ${pieceArea.toFixed(4)}`,
    )
    console.log(
      `  ${tag}: ${del.triangleCount} tris, off-level ${worst.toExponential(1)}, ` +
        `area ${(100 * relFine).toFixed(2)}% from the finer sphere, caps ${capDel.triangleCount}+${capVor.triangleCount} tris`,
    )
  }
  console.log(
    `  per slider tick: surface ${(tSurf / (2 * levels.length)).toFixed(1)} ms, ` +
      `caps ${(tCaps / (2 * levels.length)).toFixed(1)} ms (${nDirs} directions per site)`,
  )
}

// 2D: the exact regions {pi <= level} / {pi >= level} over the 3x domain
async function runCase2D(name: string, lattice: number[][], nPts: number, wMax: number, seed: number) {
  const rnd = mulberry32(seed)
  const frac = Array.from({ length: nPts }, () => [rnd(), rnd()])
  const points = frac.map((f) => lattice.map((row) => row[0] * f[0] + row[1] * f[1]))
  const weights = frac.map(() => wMax * rnd())
  const res = await fetch(`${BASE}/api/compute`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', connection: 'close' },
    body: JSON.stringify({ d: 2, lattice, points, weights, imageSize: 20 }),
  })
  if (!res.ok) throw new Error(`${name}: backend ${res.status} ${await res.text()}`)
  const r = await res.json()
  const { positions3x, kept } = r.points
  const sites: number[][] = kept.map((o: number) => positions3x[o])
  const w: number[] = kept.map((o: number) => r.points.weights[o])
  const vor: number[][] = []
  const fVor: number[] = []
  for (const a of r.voronoiGeometry.arcs) {
    vor[a.vStart] = a.start
    vor[a.vEnd] = a.end
    fVor[a.vStart] = a.fStart
    fVor[a.vEnd] = a.fEnd
  }
  const t0 = performance.now()
  const cells = buildPowerCells2D({ sites, weights: w, arcs: r.quotientArcs, basis: r.basis, vorVertices: vor })
  const pieces = buildRegionPieces2D(cells, r.domain3x.outline, 0)
  const n = cells.n
  console.log(
    `${name}: ${n} sites, ${vor.length} Voronoi vertices, ${pieces.count} pieces, ` +
      `setup ${(performance.now() - t0).toFixed(0)} ms`,
  )

  const B = r.basis as number[][]
  const shifts: number[][] = []
  for (let a = -4; a <= 4; a++)
    for (let b = -4; b <= 4; b++) shifts.push([a * B[0][0] + b * B[1][0], a * B[0][1] + b * B[1][1]])
  const zero = shifts.findIndex((s) => s[0] === 0 && s[1] === 0)
  const power = (x: number, y: number): [number, number, number] => {
    let best = Infinity
    let bi = -1
    let bs = -1
    for (let i = 0; i < n; i++)
      for (let s = 0; s < shifts.length; s++) {
        const dx = x - sites[i][0] - shifts[s][0]
        const dy = y - sites[i][1] - shifts[s][1]
        const v = dx * dx + dy * dy - w[i]
        if (v < best) {
          best = v
          bi = i
          bs = s
        }
      }
    return [best, bi, bs]
  }

  // 1. the lines describe the power cells
  let cellMismatch = 0
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 400; k++) {
      const R = cells.radius[i] * 1.1
      const y = [R * (2 * rnd() - 1), R * (2 * rnd() - 1)]
      let margin = Infinity
      for (let p = cells.lineStart[i]; p < cells.lineStart[i + 1]; p++) {
        const nn = Math.hypot(cells.lines[3 * p], cells.lines[3 * p + 1])
        margin = Math.min(
          margin,
          (cells.lines[3 * p + 2] - (cells.lines[3 * p] * y[0] + cells.lines[3 * p + 1] * y[1])) / nn,
        )
      }
      if (Math.abs(margin) < 1e-6) continue
      const [, bi, bs] = power(sites[i][0] + y[0], sites[i][1] + y[1])
      if (margin > 0 !== (bi === i && bs === zero)) cellMismatch++
    }
  }
  check(cellMismatch === 0, `cells: ${cellMismatch} sample points disagree with the brute-force nearest site`)
  let unnamed = 0
  for (let v = 0; v < cells.vertVor.length; v++) if (cells.vertVor[v] < 0) unnamed++
  check(unnamed === 0, `${unnamed} of ${cells.vertVor.length} cell vertices not matched to a Voronoi vertex`)

  // 2. the pieces tile the 3x domain
  const loop = r.domain3x.outline as number[][]
  let domainArea = 0
  for (let v = 0; v < loop.length; v++) {
    const q = loop[(v + 1) % loop.length]
    domainArea += 0.5 * (loop[v][0] * q[1] - q[0] * loop[v][1])
  }
  domainArea = Math.abs(domainArea)
  let pieceArea = 0
  for (let p = 0; p < pieces.count; p++)
    for (let v = pieces.polyStart[p]; v < pieces.polyStart[p + 1]; v++) {
      const wv = v + 1 === pieces.polyStart[p + 1] ? pieces.polyStart[p] : v + 1
      pieceArea += 0.5 * (pieces.poly[2 * v] * pieces.poly[2 * wv + 1] - pieces.poly[2 * wv] * pieces.poly[2 * v + 1])
    }
  check(Math.abs(pieceArea - domainArea) < 1e-6 * domainArea, `pieces tile the 3x domain: ${pieceArea} vs ${domainArea}`)

  // sample points of the 3x domain, for the area of {pi <= level}
  const xs = loop.map((v) => v[0])
  const ys = loop.map((v) => v[1])
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const samples: number[] = []
  while (samples.length < 60000) {
    const x = x0 + (x1 - x0) * rnd()
    const y = y0 + (y1 - y0) * rnd()
    if (r.domainA.every((a: number[], h: number) => a[0] * x + a[1] * y <= 3 * r.domainB[h])) samples.push(power(x, y)[0])
  }

  const tri2 = (P: Float32Array, o: number) =>
    0.5 * Math.abs((P[o + 3] - P[o]) * (P[o + 7] - P[o + 1]) - (P[o + 6] - P[o]) * (P[o + 4] - P[o + 1]))
  const piTop = Math.max(...fVor.map((f) => -f))
  const piBot = -Math.max(...w)
  const edgeF: number[] = r.voronoiGeometry.arcs.map((a: { filtration: number }) => a.filtration)
  const levels = [0.08, 0.3, 0.55, 0.8, 0.97].map((q) => piBot + q * (piTop - piBot))
  const fresh = edgeF.sort((a, b) => a - b)[Math.floor(edgeF.length / 3)]
  levels.push(-(fresh + 1e-4 * (piTop - piBot)))
  const sideTol = 1e-3 * (piTop - piBot)
  let tTick = 0
  for (const level of levels) {
    const tag = `level ${level.toFixed(4)}`
    const t1 = performance.now()
    const del = powerRegion2D(cells, pieces, level, false, false)
    const vorMesh = powerRegion2D(cells, pieces, level, true, true)
    tTick += performance.now() - t1

    // 3. each region on its side of the level, inside the domain
    let wrongSide = 0
    let outside = 0
    let aDel = 0
    let aVor = 0
    for (const [mesh, sign] of [[del, 1], [vorMesh, -1]] as const) {
      for (let t = 0; t < mesh.triangleCount; t++) {
        const o = 9 * t
        const a = tri2(mesh.positions, o)
        if (sign > 0) aDel += a
        else aVor += a
        if (a < 1e-10) continue
        const x = (mesh.positions[o] + mesh.positions[o + 3] + mesh.positions[o + 6]) / 3
        const y = (mesh.positions[o + 1] + mesh.positions[o + 4] + mesh.positions[o + 7]) / 3
        if (sign * (power(x, y)[0] - level) > sideTol) wrongSide++
        if (!r.domainA.every((q: number[], h: number) => q[0] * x + q[1] * y <= 3 * r.domainB[h] + 1e-6)) outside++
      }
    }
    check(wrongSide === 0, `${tag}: ${wrongSide} triangles on the wrong side of the level`)
    check(outside === 0, `${tag}: ${outside} triangles outside the 3x domain`)
    // 4. the two regions split the domain, in the sampled proportion
    check(
      Math.abs(aDel + aVor - domainArea) < 2e-3 * domainArea,
      `${tag}: areas ${aDel.toFixed(4)} + ${aVor.toFixed(4)} != domain ${domainArea.toFixed(4)}`,
    )
    const sampled = (samples.filter((v) => v <= level).length / samples.length) * domainArea
    check(Math.abs(aDel - sampled) < 0.01 * domainArea, `${tag}: disk-union area ${aDel.toFixed(4)} vs sampled ${sampled.toFixed(4)}`)

    // 5. owners: alive Voronoi vertices, one merge-tree component per connected part
    const F = -level
    const parent = Int32Array.from({ length: vor.length }, (_, v) => v)
    const find = (v: number) => {
      while (parent[v] !== v) v = parent[v] = parent[parent[v]]
      return v
    }
    for (const a of r.voronoiGeometry.arcs) if (a.filtration <= F) parent[find(a.vStart)] = find(a.vEnd)
    let badOwner = 0
    let mixed = 0
    const keyRoot = new Map<string, number>()
    for (let t = 0; t < vorMesh.triangleCount; t++) {
      if (tri2(vorMesh.positions, 9 * t) < 1e-10) continue
      const own = vorMesh.triOwner[t]
      if (own >= vor.length || !(fVor[own] <= F + 1e-9)) {
        badOwner++
        continue
      }
      const root = find(own)
      for (let c = 0; c < 3; c++) {
        const o = 9 * t + 3 * c
        const key = `${vorMesh.positions[o].toFixed(5)},${vorMesh.positions[o + 1].toFixed(5)}`
        const seen = keyRoot.get(key)
        if (seen === undefined) keyRoot.set(key, root)
        else if (seen !== root) mixed++
      }
    }
    check(badOwner === 0, `${tag}: ${badOwner} triangles owned by a missing or unborn Voronoi vertex`)
    check(mixed === 0, `${tag}: ${mixed} shared corners join parts of different components`)
    console.log(
      `  ${tag}: ${del.triangleCount}+${vorMesh.triangleCount} tris, ` +
        `disk union ${(100 * aDel / domainArea).toFixed(2)}% of the domain (sampled ${(100 * sampled / domainArea).toFixed(2)}%)`,
    )
  }
  console.log(`  per slider tick: ${(tTick / (2 * levels.length)).toFixed(2)} ms per region`)
}

await runCase2D('2D: 1 site, square', [[1, 0], [0, 1]], 1, 0, 3)
await runCase2D('2D: 5 sites, skew, weighted', [[1, 0.3], [0, 0.9]], 5, 0.05, 5)
await runCase2D('2D: 40 sites, skew, weighted', [[1, 0.3], [0, 0.9]], 40, 0.004, 9)
await runCase('1 site, cubic', LATTICES.cubic, 1, 0, 1)
await runCase('4 sites, cubic, unweighted', LATTICES.cubic, 4, 0, 2)
await runCase('8 sites, skew, weighted', LATTICES.skew, 8, 0.06, 7)
await runCase('30 sites, skew, weighted', LATTICES.skew, 30, 0.02, 11)
console.log(failures ? `${failures} check(s) FAILED` : 'All power-surface checks passed.')
process.exit(failures ? 1 : 0)
