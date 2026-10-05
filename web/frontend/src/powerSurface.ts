// Exact filtration surfaces for 3D point sets.
//
// The power distance pi(x) = min_i(|x - p_i|^2 - w_i) is quadratic inside
// each power (weighted Voronoi) cell V_i, where it equals |x - p_i|^2 - w_i.
// Its level set {pi = t} is therefore the union over the sites of the sphere
// of radius sqrt(t + w_i) around p_i clipped to V_i. That one surface bounds
// both the Delaunay ball union {pi <= t} and the Voronoi-filtration solid
// {pi >= t}, so a single construction serves both channels.
//
// Unlike marching cubes over a sampled field, nothing here depends on a
// sampling grid: the creases along the cell faces are exact, and the thin
// tubes around freshly born Voronoi edges are as thin as they really are but
// never broken up or delayed.
//
// Cells come from the periodic Delaunay quotient arcs (one bisector plane per
// incident arc), so everything is computed once per canonical site and
// instanced over the lattice by the caller.
//
// In 2D the same statement reads: inside a cell the sublevel set is the cell
// intersected with one disk, the superlevel set the cell minus that disk.
// The last section builds those regions over the 3x domain, sharing the
// polygon/disk code of the 3D domain-boundary caps.
//
// The module is three.js-free and has no runtime imports, so it can be
// exercised from node (scripts/power-surface-check.ts).

import type { IsosurfaceMesh } from './marchingCubes'

export interface PowerCellsInput {
  sites: number[][] // canonical position of each quotient (kept) site
  weights: number[] // per quotient site
  // periodic Delaunay edges: `end - start` is the offset to the neighbor copy
  arcs: { start: number[]; end: number[]; vStart: number; vEnd: number }[]
  basis: number[][] // lattice basis vectors as rows
  // one position (any lattice copy) per quotient Voronoi vertex, to name the
  // cell vertices in the Voronoi merge tree's index space; null = no naming
  vorVertices: (number[] | undefined)[] | null
  level?: number // icosphere subdivision level (default: by triangle budget)
}

export interface PowerCells {
  n: number
  centers: Float64Array // 3 per site
  weights: Float64Array
  // cell i = { y : n_k · y <= c_k } in coordinates relative to its site,
  // planes sorted nearest first; 4 numbers (nx, ny, nz, c) per plane
  planeStart: Uint32Array // n + 1 offsets
  planes: Float64Array
  // cell vertices, relative to the site
  vertStart: Uint32Array // n + 1 offsets
  verts: Float64Array // 3 per cell vertex
  vertVor: Int32Array // quotient Voronoi vertex id per cell vertex, -1 unknown
  radius: Float64Array // per site: farthest cell vertex (the ball covers the cell beyond)
  inradius: Float64Array // per site: nearest cell face (the ball is uncut below)
  basis: number[][]
  // unit icosphere shared by all sites, counterclockwise seen from outside
  dirs: Float64Array // 3 per direction
  tris: Uint32Array // 3 per triangle
  dots: Float64Array // n_k · dir, per plane then per direction
  planeNorm: Float64Array // |n_k| per plane
  // 1 - cos of the largest angle from a point of a sphere triangle to its
  // nearest corner: how far n·u can exceed its corner values inside it
  spread: number
  // Voronoi vertex reached from each (site, direction), built on first use
  dirOwner: Int32Array | null
}

function inv3(m: number[][]): number[][] {
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

// Subdivision level keeping (sites × sphere triangles) within a budget: the
// mesh is instanced over every lattice translate covering the 3x domain.
const TRIANGLE_BUDGET = 120_000
export function icoLevelFor(nSites: number): number {
  let level = 5
  while (level > 1 && nSites * 20 * 4 ** level > TRIANGLE_BUDGET) level--
  return level
}

function icosphere(level: number): { dirs: Float64Array; tris: Uint32Array } {
  const t = (1 + Math.sqrt(5)) / 2
  const verts: number[] = [
    -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, t, 0, -1, t, 0,
    1, -t, 0, -1, -t, 0, 1,
  ]
  let faces: number[] = [
    0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1,
    8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
  ]
  const normalize = (i: number) => {
    const l = Math.hypot(verts[3 * i], verts[3 * i + 1], verts[3 * i + 2])
    verts[3 * i] /= l
    verts[3 * i + 1] /= l
    verts[3 * i + 2] /= l
  }
  for (let i = 0; i < 12; i++) normalize(i)
  for (let s = 0; s < level; s++) {
    const mid = new Map<number, number>()
    const midpoint = (a: number, b: number) => {
      const key = a < b ? a * 0x4000000 + b : b * 0x4000000 + a
      let m = mid.get(key)
      if (m === undefined) {
        m = verts.length / 3
        verts.push(
          verts[3 * a] + verts[3 * b],
          verts[3 * a + 1] + verts[3 * b + 1],
          verts[3 * a + 2] + verts[3 * b + 2],
        )
        normalize(m)
        mid.set(key, m)
      }
      return m
    }
    const next: number[] = []
    for (let f = 0; f < faces.length; f += 3) {
      const a = faces[f]
      const b = faces[f + 1]
      const c = faces[f + 2]
      const ab = midpoint(a, b)
      const bc = midpoint(b, c)
      const ca = midpoint(c, a)
      next.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca)
    }
    faces = next
  }
  return { dirs: Float64Array.from(verts), tris: Uint32Array.from(faces) }
}

// Vertices of the bounded cell { y : n_k · y <= c_k }: every triple of planes
// meeting at a point that satisfies all the others. `pl` holds 4 numbers per
// plane; `len` is the cell's length scale for the tolerances.
function cellVertices(pl: number[], len: number): number[] {
  const K = pl.length / 4
  const out: number[] = []
  const dedupe2 = (1e-7 * len) ** 2
  for (let j = 0; j < K; j++) {
    const j0 = 4 * j
    for (let k = j + 1; k < K; k++) {
      const k0 = 4 * k
      // n_j × n_k
      const ex = pl[j0 + 1] * pl[k0 + 2] - pl[j0 + 2] * pl[k0 + 1]
      const ey = pl[j0 + 2] * pl[k0] - pl[j0] * pl[k0 + 2]
      const ez = pl[j0] * pl[k0 + 1] - pl[j0 + 1] * pl[k0]
      const nj = Math.hypot(pl[j0], pl[j0 + 1], pl[j0 + 2])
      const nk = Math.hypot(pl[k0], pl[k0 + 1], pl[k0 + 2])
      if (Math.hypot(ex, ey, ez) < 1e-9 * nj * nk) continue
      for (let l = k + 1; l < K; l++) {
        const l0 = 4 * l
        const nl = Math.hypot(pl[l0], pl[l0 + 1], pl[l0 + 2])
        const det = ex * pl[l0] + ey * pl[l0 + 1] + ez * pl[l0 + 2]
        if (Math.abs(det) < 1e-9 * nj * nk * nl) continue
        // x = (c_j (n_k × n_l) + c_k (n_l × n_j) + c_l (n_j × n_k)) / det
        const ax = pl[k0 + 1] * pl[l0 + 2] - pl[k0 + 2] * pl[l0 + 1]
        const ay = pl[k0 + 2] * pl[l0] - pl[k0] * pl[l0 + 2]
        const az = pl[k0] * pl[l0 + 1] - pl[k0 + 1] * pl[l0]
        const bx = pl[l0 + 1] * pl[j0 + 2] - pl[l0 + 2] * pl[j0 + 1]
        const by = pl[l0 + 2] * pl[j0] - pl[l0] * pl[j0 + 2]
        const bz = pl[l0] * pl[j0 + 1] - pl[l0 + 1] * pl[j0]
        const x = (pl[j0 + 3] * ax + pl[k0 + 3] * bx + pl[l0 + 3] * ex) / det
        const y = (pl[j0 + 3] * ay + pl[k0 + 3] * by + pl[l0 + 3] * ey) / det
        const z = (pl[j0 + 3] * az + pl[k0 + 3] * bz + pl[l0 + 3] * ez) / det
        let inside = true
        for (let m = 0; m < K && inside; m++) {
          if (m === j || m === k || m === l) continue
          const m0 = 4 * m
          const nm = Math.hypot(pl[m0], pl[m0 + 1], pl[m0 + 2])
          const viol = pl[m0] * x + pl[m0 + 1] * y + pl[m0 + 2] * z - pl[m0 + 3]
          if (viol > 1e-9 * (nm * len + Math.abs(pl[m0 + 3]))) inside = false
        }
        if (!inside) continue
        let dup = false
        for (let v = 0; v < out.length && !dup; v += 3) {
          const dx = out[v] - x
          const dy = out[v + 1] - y
          const dz = out[v + 2] - z
          if (dx * dx + dy * dy + dz * dz < dedupe2) dup = true
        }
        if (!dup) out.push(x, y, z)
      }
    }
  }
  return out
}

export function buildPowerCells(input: PowerCellsInput): PowerCells {
  const n = input.sites.length
  const centers = new Float64Array(3 * n)
  const weights = Float64Array.from(input.weights)
  for (let i = 0; i < n; i++) {
    centers[3 * i] = input.sites[i][0]
    centers[3 * i + 1] = input.sites[i][1]
    centers[3 * i + 2] = input.sites[i][2]
  }

  // bisector planes: power_i(y) <= power_j(y) for the neighbor at offset d
  // is 2 d · y <= |d|^2 + w_i - w_j
  const lists: number[][] = Array.from({ length: n }, () => [])
  for (const a of input.arcs) {
    const dx = a.end[0] - a.start[0]
    const dy = a.end[1] - a.start[1]
    const dz = a.end[2] - a.start[2]
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 === 0) continue
    const ws = weights[a.vStart]
    const we = weights[a.vEnd]
    lists[a.vStart].push(2 * dx, 2 * dy, 2 * dz, d2 + ws - we)
    lists[a.vEnd].push(-2 * dx, -2 * dy, -2 * dz, d2 + we - ws)
  }

  const planeStart = new Uint32Array(n + 1)
  const vertStart = new Uint32Array(n + 1)
  const radius = new Float64Array(n)
  const inradius = new Float64Array(n)
  const planeChunks: number[][] = []
  const vertChunks: number[][] = []
  for (let i = 0; i < n; i++) {
    const raw = lists[i]
    const K = raw.length / 4
    // nearest planes first: the trivial-reject mask covers the first 31
    const order = Array.from({ length: K }, (_, k) => k).sort((p, q) => {
      const dp = raw[4 * p + 3] / Math.hypot(raw[4 * p], raw[4 * p + 1], raw[4 * p + 2])
      const dq = raw[4 * q + 3] / Math.hypot(raw[4 * q], raw[4 * q + 1], raw[4 * q + 2])
      return dp - dq
    })
    const pl: number[] = []
    let len = 0
    let rin = Infinity
    for (const k of order) {
      pl.push(raw[4 * k], raw[4 * k + 1], raw[4 * k + 2], raw[4 * k + 3])
      const nn = Math.hypot(raw[4 * k], raw[4 * k + 1], raw[4 * k + 2])
      len = Math.max(len, nn / 2)
      rin = Math.min(rin, raw[4 * k + 3] / nn)
    }
    const verts = K >= 4 ? cellVertices(pl, len) : []
    let r = 0
    for (let v = 0; v < verts.length; v += 3)
      r = Math.max(r, Math.hypot(verts[v], verts[v + 1], verts[v + 2]))
    radius[i] = r
    inradius[i] = Math.max(0, Number.isFinite(rin) ? rin : 0)
    planeChunks.push(pl)
    vertChunks.push(verts)
    planeStart[i + 1] = planeStart[i] + K
    vertStart[i + 1] = vertStart[i] + verts.length / 3
  }
  const planes = Float64Array.from(planeChunks.flat())
  const verts = Float64Array.from(vertChunks.flat())

  // name the cell vertices: nearest quotient Voronoi vertex modulo the lattice
  const vertVor = new Int32Array(verts.length / 3).fill(-1)
  const vor = input.vorVertices
  if (vor) {
    const B = input.basis
    // columns of M are the basis vectors: x = M z
    const M = [
      [B[0][0], B[1][0], B[2][0]],
      [B[0][1], B[1][1], B[2][1]],
      [B[0][2], B[1][2], B[2][2]],
    ]
    const Mi = inv3(M)
    const scale = Math.hypot(B[0][0], B[0][1], B[0][2])
    const tol2 = (1e-5 * scale) ** 2
    for (let i = 0; i < n; i++) {
      for (let v = vertStart[i]; v < vertStart[i + 1]; v++) {
        const X = centers[3 * i] + verts[3 * v]
        const Y = centers[3 * i + 1] + verts[3 * v + 1]
        const Z = centers[3 * i + 2] + verts[3 * v + 2]
        let best = -1
        let bestD = Infinity
        for (let id = 0; id < vor.length; id++) {
          const R = vor[id]
          if (!R) continue
          const dx = X - R[0]
          const dy = Y - R[1]
          const dz = Z - R[2]
          let f0 = Mi[0][0] * dx + Mi[0][1] * dy + Mi[0][2] * dz
          let f1 = Mi[1][0] * dx + Mi[1][1] * dy + Mi[1][2] * dz
          let f2 = Mi[2][0] * dx + Mi[2][1] * dy + Mi[2][2] * dz
          f0 -= Math.round(f0)
          f1 -= Math.round(f1)
          f2 -= Math.round(f2)
          const rx = M[0][0] * f0 + M[0][1] * f1 + M[0][2] * f2
          const ry = M[1][0] * f0 + M[1][1] * f1 + M[1][2] * f2
          const rz = M[2][0] * f0 + M[2][1] * f1 + M[2][2] * f2
          const d2 = rx * rx + ry * ry + rz * rz
          if (d2 < bestD) {
            bestD = d2
            best = id
          }
        }
        if (bestD < tol2) vertVor[v] = best
      }
    }
  }

  const { dirs, tris } = icosphere(input.level ?? icoLevelFor(n))
  const nDirs = dirs.length / 3
  const nPlanes = planes.length / 4
  const dots = new Float64Array(nPlanes * nDirs)
  for (let p = 0; p < nPlanes; p++) {
    const nx = planes[4 * p]
    const ny = planes[4 * p + 1]
    const nz = planes[4 * p + 2]
    const row = p * nDirs
    for (let v = 0; v < nDirs; v++)
      dots[row + v] = nx * dirs[3 * v] + ny * dirs[3 * v + 1] + nz * dirs[3 * v + 2]
  }

  const planeNorm = new Float64Array(nPlanes)
  for (let p = 0; p < nPlanes; p++)
    planeNorm[p] = Math.hypot(planes[4 * p], planes[4 * p + 1], planes[4 * p + 2])
  // circumradius of the widest sphere triangle
  let cosMin = 1
  for (let f = 0; f < tris.length; f += 3) {
    const a = 3 * tris[f]
    const b = 3 * tris[f + 1]
    const c = 3 * tris[f + 2]
    const ux = dirs[b] - dirs[a]
    const uy = dirs[b + 1] - dirs[a + 1]
    const uz = dirs[b + 2] - dirs[a + 2]
    const vx = dirs[c] - dirs[a]
    const vy = dirs[c + 1] - dirs[a + 1]
    const vz = dirs[c + 2] - dirs[a + 2]
    const nx = uy * vz - uz * vy
    const ny = uz * vx - ux * vz
    const nz = ux * vy - uy * vx
    const cos = Math.abs(nx * dirs[a] + ny * dirs[a + 1] + nz * dirs[a + 2]) / Math.hypot(nx, ny, nz)
    cosMin = Math.min(cosMin, cos)
  }

  return {
    n,
    centers,
    weights,
    planeStart,
    planes,
    vertStart,
    verts,
    vertVor,
    radius,
    inradius,
    basis: input.basis,
    dirs,
    tris,
    dots,
    planeNorm,
    spread: 1 - cosMin,
    dirOwner: null,
  }
}

// ---- linking surface patches to the Voronoi merge tree ----------------------

// The quotient Voronoi vertex whose component owns the point y of cell i
// (y relative to the site): walk away from the site along the ray through y
// to a face, then away from the site's foot on that face to an edge, then
// away from its foot on that edge to a vertex. The power distance to the
// site never decreases along the walk and the walk stays in the cell, so it
// stays inside the superlevel set of any level y belongs to — the vertex
// reached is alive and in y's component. Independent of the level.
export function ascend(cells: PowerCells, i: number, x: number, y: number, z: number): number {
  const { planes, planeStart, verts, vertStart, vertVor } = cells
  const p0 = planeStart[i]
  const p1 = planeStart[i + 1]
  if (p1 - p0 < 4 || vertStart[i + 1] === vertStart[i]) return -1

  // 1. radial exit through face j
  let j = -1
  let s = Infinity
  for (let p = p0; p < p1; p++) {
    const d = planes[4 * p] * x + planes[4 * p + 1] * y + planes[4 * p + 2] * z
    if (d > 0) {
      const sp = planes[4 * p + 3] / d
      if (sp < s) {
        s = sp
        j = p
      }
    }
  }
  let vx = x
  let vy = y
  let vz = z
  if (j >= 0) {
    if (s < 1) s = 1
    vx = s * x
    vy = s * y
    vz = s * z
    const jx = planes[4 * j]
    const jy = planes[4 * j + 1]
    const jz = planes[4 * j + 2]
    const jn2 = jx * jx + jy * jy + jz * jz
    // 2. on face j, away from the foot of the site
    const fs = planes[4 * j + 3] / jn2
    let dx = vx - fs * jx
    let dy = vy - fs * jy
    let dz = vz - fs * jz
    if (dx * dx + dy * dy + dz * dz < 1e-24 * jn2) {
      // the ray hit the foot itself: any in-plane direction climbs
      if (Math.abs(jx) < Math.abs(jy)) {
        dx = 0
        dy = -jz
        dz = jy
      } else {
        dx = jz
        dy = 0
        dz = -jx
      }
    }
    const dn = Math.hypot(dx, dy, dz)
    let k = -1
    let u = Infinity
    for (let p = p0; p < p1; p++) {
      if (p === j) continue
      const px = planes[4 * p]
      const py = planes[4 * p + 1]
      const pz = planes[4 * p + 2]
      const den = px * dx + py * dy + pz * dz
      if (den > 1e-12 * Math.hypot(px, py, pz) * dn) {
        const up = (planes[4 * p + 3] - (px * vx + py * vy + pz * vz)) / den
        if (up < u) {
          u = up
          k = p
        }
      }
    }
    if (k >= 0) {
      if (u < 0) u = 0
      vx += u * dx
      vy += u * dy
      vz += u * dz
      // 3. on edge j ∩ k, away from the foot of the site
      const kx = planes[4 * k]
      const ky = planes[4 * k + 1]
      const kz = planes[4 * k + 2]
      let ex = jy * kz - jz * ky
      let ey = jz * kx - jx * kz
      let ez = jx * ky - jy * kx
      const en = Math.hypot(ex, ey, ez)
      if (en > 0) {
        if (vx * ex + vy * ey + vz * ez < 0) {
          ex = -ex
          ey = -ey
          ez = -ez
        }
        let tau = Infinity
        for (let p = p0; p < p1; p++) {
          if (p === j || p === k) continue
          const px = planes[4 * p]
          const py = planes[4 * p + 1]
          const pz = planes[4 * p + 2]
          const den = px * ex + py * ey + pz * ez
          if (den > 1e-12 * Math.hypot(px, py, pz) * en) {
            const tp = (planes[4 * p + 3] - (px * vx + py * vy + pz * vz)) / den
            if (tp < tau) tau = tp
          }
        }
        if (Number.isFinite(tau)) {
          if (tau < 0) tau = 0
          vx += tau * ex
          vy += tau * ey
          vz += tau * ez
        }
      }
    }
  }

  let best = -1
  let bestD = Infinity
  for (let v = vertStart[i]; v < vertStart[i + 1]; v++) {
    const dx = verts[3 * v] - vx
    const dy = verts[3 * v + 1] - vy
    const dz = verts[3 * v + 2] - vz
    const d2 = dx * dx + dy * dy + dz * dz
    if (d2 < bestD) {
      bestD = d2
      best = v
    }
  }
  return best < 0 ? -1 : vertVor[best]
}

function ensureDirOwners(cells: PowerCells): Int32Array {
  if (cells.dirOwner) return cells.dirOwner
  const nDirs = cells.dirs.length / 3
  const out = new Int32Array(cells.n * nDirs)
  for (let i = 0; i < cells.n; i++)
    for (let v = 0; v < nDirs; v++)
      out[i * nDirs + v] = ascend(
        cells,
        i,
        cells.dirs[3 * v],
        cells.dirs[3 * v + 1],
        cells.dirs[3 * v + 2],
      )
  cells.dirOwner = out
  return out
}

// ---- the surface (once per threshold) ---------------------------------------

// Growable unindexed triangle soup in the IsosurfaceMesh layout.
function meshWriter(capacity: number) {
  let cap = Math.max(256, capacity)
  let positions = new Float32Array(cap * 9)
  let normals = new Float32Array(cap * 9)
  let owners = new Uint32Array(cap)
  let count = 0
  return {
    reserve(extra: number) {
      if (count + extra <= cap) return
      while (cap < count + extra) cap *= 2
      const p = new Float32Array(cap * 9)
      p.set(positions.subarray(0, count * 9))
      positions = p
      const q = new Float32Array(cap * 9)
      q.set(normals.subarray(0, count * 9))
      normals = q
      const o = new Uint32Array(cap)
      o.set(owners.subarray(0, count))
      owners = o
    },
    // one triangle; the caller has reserved room
    tri(
      x0: number, y0: number, z0: number, x1: number, y1: number, z1: number,
      x2: number, y2: number, z2: number,
      a0: number, b0: number, c0: number, a1: number, b1: number, c1: number,
      a2: number, b2: number, c2: number,
      owner: number,
    ) {
      const o = count * 9
      positions[o] = x0
      positions[o + 1] = y0
      positions[o + 2] = z0
      positions[o + 3] = x1
      positions[o + 4] = y1
      positions[o + 5] = z1
      positions[o + 6] = x2
      positions[o + 7] = y2
      positions[o + 8] = z2
      normals[o] = a0
      normals[o + 1] = b0
      normals[o + 2] = c0
      normals[o + 3] = a1
      normals[o + 4] = b1
      normals[o + 5] = c1
      normals[o + 6] = a2
      normals[o + 7] = b2
      normals[o + 8] = c2
      owners[count++] = owner
    },
    finish(): IsosurfaceMesh {
      return {
        positions: positions.subarray(0, count * 9),
        normals: normals.subarray(0, count * 9),
        triOwner: owners.subarray(0, count),
        triangleCount: count,
      }
    },
  }
}

// Sphere triangles cut by a cell face are subdivided this many times before
// clipping, so the creases are drawn (and thin strips resolved) 4x finer
// than the sphere tessellation.
const REFINE_DEPTH = 2

// The level set {pi = level} over one period: for each site the sphere of
// radius sqrt(level + w_i) cut to its cell. Sphere triangles clear of the
// cell faces are emitted as they are; the ones a face may cut are refined
// and clipped ON the sphere — each new corner is where the great arc between
// two corners meets the face — so every vertex lies on the level set and a
// strip of surface squeezed between two faces keeps its true width however
// thin it is. Normals are analytic.
// `voronoi` selects the side: false = boundary of the ball union (normals
// outward from the balls, triOwner = site), true = boundary of the
// Voronoi-filtration solid outside the balls (normals into the balls,
// triOwner = a quotient Voronoi vertex of the bounded component; only
// computed when `withOwners`).
export function powerSurface(
  cells: PowerCells,
  level: number,
  voronoi: boolean,
  withOwners: boolean,
): IsosurfaceMesh {
  const { n, centers, weights, planeStart, planes, planeNorm, dirs, tris, dots } = cells
  const nDirs = dirs.length / 3
  const nTris = tris.length / 3
  const out = meshWriter(Math.min(n * nTris, 1 << 16))
  const dirOwner = voronoi && withOwners ? ensureDirOwners(cells) : null
  const sgn = voronoi ? -1 : 1

  const near = new Uint8Array(nDirs) // not clear of every face
  const mask = new Int32Array(nDirs) // faces (first 31) the direction is beyond
  // clip polygon scratch (a triangle gains at most one vertex per plane)
  let maxK = 0
  for (let i = 0; i < n; i++) maxK = Math.max(maxK, planeStart[i + 1] - planeStart[i])
  let px = new Float64Array(maxK + 4)
  let py = new Float64Array(maxK + 4)
  let pz = new Float64Array(maxK + 4)
  let qx = new Float64Array(maxK + 4)
  let qy = new Float64Array(maxK + 4)
  let qz = new Float64Array(maxK + 4)
  const dist = new Float64Array(maxK + 4)

  // the site being processed
  let si = 0
  let r = 0
  let r2 = 0
  let p0 = 0
  let p1 = 0
  let cx = 0
  let cy = 0
  let cz = 0

  // owner of the patch around the unit direction (x, y, z)
  const ownerAt = (x: number, y: number, z: number) => (dirOwner ? ascend(cells, si, x, y, z) : si)

  // one sphere triangle of the current site, corners as unit directions
  const whole = (
    ax: number, ay: number, az: number, bx: number, by: number, bz: number,
    dx: number, dy: number, dz: number, owner: number,
  ) => {
    out.reserve(1)
    out.tri(
      cx + r * ax, cy + r * ay, cz + r * az,
      cx + r * bx, cy + r * by, cz + r * bz,
      cx + r * dx, cy + r * dy, cz + r * dz,
      sgn * ax, sgn * ay, sgn * az,
      sgn * bx, sgn * by, sgn * bz,
      sgn * dx, sgn * dy, sgn * dz,
      owner,
    )
  }

  // clip one spherical triangle against the cell (Sutherland–Hodgman with
  // the crossings taken along the great arcs) and emit what is left
  const clip = (
    ax: number, ay: number, az: number, bx: number, by: number, bz: number,
    dx: number, dy: number, dz: number,
  ) => {
    let m = 3
    px[0] = r * ax
    py[0] = r * ay
    pz[0] = r * az
    px[1] = r * bx
    py[1] = r * by
    pz[1] = r * bz
    px[2] = r * dx
    py[2] = r * dy
    pz[2] = r * dz
    for (let p = p0; p < p1 && m >= 3; p++) {
      const nx = planes[4 * p]
      const ny = planes[4 * p + 1]
      const nz = planes[4 * p + 2]
      const pc = planes[4 * p + 3]
      let anyOut = false
      let anyIn = false
      for (let v = 0; v < m; v++) {
        const d = nx * px[v] + ny * py[v] + nz * pz[v] - pc
        dist[v] = d
        if (d > 0) anyOut = true
        else anyIn = true
      }
      if (!anyOut) continue
      if (!anyIn) return
      let k = 0
      for (let v = 0; v < m; v++) {
        const w = v + 1 === m ? 0 : v + 1
        const dv = dist[v]
        const dw = dist[w]
        if (dv <= 0) {
          qx[k] = px[v]
          qy[k] = py[v]
          qz[k++] = pz[v]
        }
        if (dv <= 0 !== dw <= 0) {
          // the point of the arc v→w on the plane: with P(s) the chord
          // point, r n·P(s) = c |P(s)|, squared into a quadratic in s
          const al = dv + pc
          const be = dw - dv
          const bq = px[v] * px[w] + py[v] * py[w] + pz[v] * pz[w] - r2
          const qa = r2 * be * be + 2 * pc * pc * bq
          const qb = 2 * r2 * al * be - 2 * pc * pc * bq
          const qc = r2 * (al * al - pc * pc)
          let s = dv / (dv - dw) // chord crossing: the fallback
          const disc = qb * qb - 4 * qa * qc
          if (disc >= 0 && qa !== 0) {
            const sq = Math.sqrt(disc)
            const h = -0.5 * (qb + (qb >= 0 ? sq : -sq))
            const s1 = h / qa
            const s2 = h !== 0 ? qc / h : s1
            // the root inside the arc; of two, the one nearer the chord's
            const ok1 = s1 >= 0 && s1 <= 1
            const ok2 = s2 >= 0 && s2 <= 1
            if (ok1 && ok2) s = Math.abs(s1 - s) <= Math.abs(s2 - s) ? s1 : s2
            else if (ok1) s = s1
            else if (ok2) s = s2
          }
          const wx = px[v] + s * (px[w] - px[v])
          const wy = py[v] + s * (py[w] - py[v])
          const wz = pz[v] + s * (pz[w] - pz[v])
          const l = r / Math.hypot(wx, wy, wz)
          qx[k] = wx * l
          qy[k] = wy * l
          qz[k++] = wz * l
        }
      }
      m = k
      let tmp = px
      px = qx
      qx = tmp
      tmp = py
      py = qy
      qy = tmp
      tmp = pz
      pz = qz
      qz = tmp
    }
    if (m < 3) return
    let mx = 0
    let my = 0
    let mz = 0
    for (let v = 0; v < m; v++) {
      mx += px[v]
      my += py[v]
      mz += pz[v]
    }
    const owner = ownerAt(mx, my, mz)
    out.reserve(m - 2)
    const ir = sgn / r
    for (let v = 1; v + 1 < m; v++) {
      out.tri(
        cx + px[0], cy + py[0], cz + pz[0],
        cx + px[v], cy + py[v], cz + pz[v],
        cx + px[v + 1], cy + py[v + 1], cz + pz[v + 1],
        ir * px[0], ir * py[0], ir * pz[0],
        ir * px[v], ir * py[v], ir * pz[v],
        ir * px[v + 1], ir * py[v + 1], ir * pz[v + 1],
        owner,
      )
    }
  }

  // A sphere triangle some face may cut. `spread` bounds how far n·u can
  // exceed its corner values inside the triangle (a face can poke through
  // the middle without reaching a corner), so only triangles clear of every
  // face by that margin are taken whole; the rest are split until the
  // margin is negligible, then clipped.
  const refine = (
    ax: number, ay: number, az: number, bx: number, by: number, bz: number,
    dx: number, dy: number, dz: number, depth: number, spread: number,
  ) => {
    let clear = true
    for (let p = p0; p < p1; p++) {
      const nx = planes[4 * p]
      const ny = planes[4 * p + 1]
      const nz = planes[4 * p + 2]
      const lim = planes[4 * p + 3] / r
      const da = nx * ax + ny * ay + nz * az - lim
      const db = nx * bx + ny * by + nz * bz - lim
      const dd = nx * dx + ny * dy + nz * dz - lim
      // all corners beyond one face (a convex cap when it is on the site's side)
      if (da > 0 && db > 0 && dd > 0 && lim >= 0) return
      const margin = depth > 0 ? -planeNorm[p] * spread : 0
      if (da > margin || db > margin || dd > margin) clear = false
    }
    if (clear) {
      whole(ax, ay, az, bx, by, bz, dx, dy, dz, ownerAt(ax + bx + dx, ay + by + dy, az + bz + dz))
      return
    }
    if (depth === 0) {
      clip(ax, ay, az, bx, by, bz, dx, dy, dz)
      return
    }
    let ex = ax + bx
    let ey = ay + by
    let ez = az + bz
    let l = 1 / Math.hypot(ex, ey, ez)
    ex *= l
    ey *= l
    ez *= l
    let fx = bx + dx
    let fy = by + dy
    let fz = bz + dz
    l = 1 / Math.hypot(fx, fy, fz)
    fx *= l
    fy *= l
    fz *= l
    let gx = dx + ax
    let gy = dy + ay
    let gz = dz + az
    l = 1 / Math.hypot(gx, gy, gz)
    gx *= l
    gy *= l
    gz *= l
    const sub = spread / 4
    refine(ax, ay, az, ex, ey, ez, gx, gy, gz, depth - 1, sub)
    refine(bx, by, bz, fx, fy, fz, ex, ey, ez, depth - 1, sub)
    refine(dx, dy, dz, gx, gy, gz, fx, fy, fz, depth - 1, sub)
    refine(ex, ey, ez, fx, fy, fz, gx, gy, gz, depth - 1, sub)
  }

  for (let i = 0; i < n; i++) {
    r2 = level + weights[i]
    if (!(r2 > 0)) continue
    r = Math.sqrt(r2)
    if (r >= cells.radius[i]) continue // the ball covers the whole cell
    si = i
    p0 = planeStart[i]
    p1 = planeStart[i + 1]
    cx = centers[3 * i]
    cy = centers[3 * i + 1]
    cz = centers[3 * i + 2]
    const ownBase = i * nDirs

    const uncut = r <= cells.inradius[i]
    if (!uncut) {
      near.fill(0)
      mask.fill(0)
      for (let p = p0; p < p1; p++) {
        const lim = planes[4 * p + 3] / r
        const margin = lim - planeNorm[p] * cells.spread
        const row = p * nDirs
        const bit = p - p0 < 31 && lim >= 0 ? 1 << (p - p0) : 0
        for (let v = 0; v < nDirs; v++) {
          const d = dots[row + v]
          if (d > margin) {
            near[v] = 1
            if (d > lim) mask[v] |= bit
          }
        }
      }
    }

    out.reserve(nTris)
    for (let f = 0; f < nTris; f++) {
      const a = tris[3 * f]
      const b = voronoi ? tris[3 * f + 2] : tris[3 * f + 1]
      const c = voronoi ? tris[3 * f + 1] : tris[3 * f + 2]
      if (uncut || (!near[a] && !near[b] && !near[c])) {
        whole(
          dirs[3 * a], dirs[3 * a + 1], dirs[3 * a + 2],
          dirs[3 * b], dirs[3 * b + 1], dirs[3 * b + 2],
          dirs[3 * c], dirs[3 * c + 1], dirs[3 * c + 2],
          dirOwner ? dirOwner[ownBase + a] : i,
        )
        continue
      }
      // all three corners beyond one common face: nothing of it is inside
      if (mask[a] & mask[b] & mask[c]) continue
      refine(
        dirs[3 * a], dirs[3 * a + 1], dirs[3 * a + 2],
        dirs[3 * b], dirs[3 * b + 1], dirs[3 * b + 2],
        dirs[3 * c], dirs[3 * c + 1], dirs[3 * c + 2],
        REFINE_DEPTH,
        cells.spread,
      )
    }
  }
  return out.finish()
}

// ---- lattice translates covering the 3x domain -------------------------------

// Lattice vectors T for which some cell copy V_i + p_i + T can reach into the
// 3x Dirichlet domain {A x <= 3b}: per domain face, the cell's extreme vertex
// must not be beyond it (separating-axis test on the face normals; the
// caller's clipping planes cut away the rest).
export function surfaceTranslates(
  cells: PowerCells,
  domainA: number[][],
  domainB: number[],
  domainVertices: number[][],
): [number, number, number][] {
  const { n, centers, verts, vertStart, basis } = cells
  const H = domainA.length
  // per site and face: A_h · p_i + min over its cell vertices of A_h · v
  const reach = new Float64Array(n * H)
  let maxR = 0
  for (let i = 0; i < n; i++) {
    maxR = Math.max(
      maxR,
      cells.radius[i] + Math.hypot(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]),
    )
    for (let h = 0; h < H; h++) {
      const a = domainA[h]
      let lo = Infinity
      for (let v = vertStart[i]; v < vertStart[i + 1]; v++)
        lo = Math.min(lo, a[0] * verts[3 * v] + a[1] * verts[3 * v + 1] + a[2] * verts[3 * v + 2])
      reach[i * H + h] =
        a[0] * centers[3 * i] + a[1] * centers[3 * i + 1] + a[2] * centers[3 * i + 2] + lo
    }
  }
  let R = 0
  for (const v of domainVertices) R = Math.max(R, Math.hypot(v[0], v[1], v[2]))
  // T = Σ z_k basis[k] ⇒ |z_k| <= |row k of the inverse| · |T|
  const Bi = inv3([
    [basis[0][0], basis[1][0], basis[2][0]],
    [basis[0][1], basis[1][1], basis[2][1]],
    [basis[0][2], basis[1][2], basis[2][2]],
  ])
  const zMax = [0, 1, 2].map((k) => Math.ceil(Math.hypot(Bi[k][0], Bi[k][1], Bi[k][2]) * (R + maxR)))
  const out: [number, number, number][] = []
  for (let z0 = -zMax[0]; z0 <= zMax[0]; z0++)
    for (let z1 = -zMax[1]; z1 <= zMax[1]; z1++)
      for (let z2 = -zMax[2]; z2 <= zMax[2]; z2++) {
        const tx = z0 * basis[0][0] + z1 * basis[1][0] + z2 * basis[2][0]
        const ty = z0 * basis[0][1] + z1 * basis[1][1] + z2 * basis[2][1]
        const tz = z0 * basis[0][2] + z1 * basis[1][2] + z2 * basis[2][2]
        let any = false
        for (let i = 0; i < n && !any; i++) {
          if (vertStart[i + 1] === vertStart[i]) continue
          let ok = true
          for (let h = 0; h < H && ok; h++) {
            const a = domainA[h]
            if (a[0] * tx + a[1] * ty + a[2] * tz + reach[i * H + h] > 3 * domainB[h]) ok = false
          }
          any = ok
        }
        if (any) out.push([tx, ty, tz])
      }
  return out
}

// ---- domain-boundary caps (solid rendering) ----------------------------------

// The 3x domain facets cut into pieces by the power cells crossing them: on
// a piece the nearest site is known, so the solid's cross-section there is
// the piece intersected with (Delaunay) or minus (Voronoi) one disk.
export interface CapPieces {
  count: number
  site: Uint32Array // quotient site of each piece
  center: Float64Array // 3 per piece: position of the site copy
  facet: Uint32Array // facet of each piece
  // per facet: origin, in-plane axes e1, e2 and outward normal (e1 × e2 = n)
  frames: Float64Array // 12 per facet
  polyStart: Uint32Array // count + 1 offsets
  poly: Float64Array // 2 per vertex: (u, v) in the facet frame, counterclockwise
  o: Float64Array // 2 per piece: the site copy projected into the facet frame
  h2: Float64Array // squared distance of the site copy to the facet plane
  rhoMin: Float64Array // nearest / farthest point of the piece from o
  rhoMax: Float64Array
  wholeOwner: Int32Array // cached Voronoi owner of the uncut piece, -2 = not yet
}

// Clip the convex polygon (xs, ys)[0..m) to a·x + b·y <= c; returns the new
// count, result in (ox, oy).
function clip2(
  xs: Float64Array, ys: Float64Array, m: number,
  a: number, b: number, c: number,
  ox: Float64Array, oy: Float64Array,
): number {
  let k = 0
  for (let v = 0; v < m; v++) {
    const w = v + 1 === m ? 0 : v + 1
    const dv = a * xs[v] + b * ys[v] - c
    const dw = a * xs[w] + b * ys[w] - c
    if (dv <= 0) {
      ox[k] = xs[v]
      oy[k++] = ys[v]
    }
    if (dv <= 0 !== dw <= 0) {
      const s = dv / (dv - dw)
      ox[k] = xs[v] + s * (xs[w] - xs[v])
      oy[k++] = ys[v] + s * (ys[w] - ys[v])
    }
  }
  return k
}

// Drop consecutive vertices closer than sqrt(eps2), in place; returns the new
// count. Clipping at a corner leaves near-coincident vertices whose joining
// edge has a meaningless direction — poison for any halfplane test on it.
function dedupe2(xs: Float64Array, ys: Float64Array, m: number, eps2: number): number {
  let k = 0
  for (let v = 0; v < m; v++) {
    if (k > 0 && (xs[v] - xs[k - 1]) ** 2 + (ys[v] - ys[k - 1]) ** 2 < eps2) continue
    xs[k] = xs[v]
    ys[k++] = ys[v]
  }
  while (k > 1 && (xs[0] - xs[k - 1]) ** 2 + (ys[0] - ys[k - 1]) ** 2 < eps2) k--
  return k
}

// Nearest (0 when inside) and farthest distance from (su, sv) to the
// counterclockwise convex polygon (xs, ys)[0..m).
function pieceReach(
  xs: Float64Array, ys: Float64Array, m: number, su: number, sv: number,
): [number, number] {
  let rmax = 0
  let rmin = Infinity
  let inside = true
  for (let v = 0; v < m; v++) {
    const w = v + 1 === m ? 0 : v + 1
    rmax = Math.max(rmax, Math.hypot(xs[v] - su, ys[v] - sv))
    const ex = xs[w] - xs[v]
    const ey = ys[w] - ys[v]
    if (ex * (sv - ys[v]) - ey * (su - xs[v]) < 0) inside = false
    const el = ex * ex + ey * ey
    let tt = el > 0 ? ((su - xs[v]) * ex + (sv - ys[v]) * ey) / el : 0
    tt = Math.max(0, Math.min(1, tt))
    rmin = Math.min(rmin, Math.hypot(xs[v] + tt * ex - su, ys[v] + tt * ey - sv))
  }
  return [inside ? 0 : rmin, rmax]
}

export function buildCapPieces(
  cells: PowerCells,
  domainA: number[][],
  domainB: number[],
  domainVertices: number[][],
): CapPieces {
  const { n, centers, planes, planeStart, basis } = cells
  const H = domainA.length
  const frames = new Float64Array(12 * H)
  const site: number[] = []
  const center: number[] = []
  const facet: number[] = []
  const polyStart: number[] = [0]
  const poly: number[] = []
  const o: number[] = []
  const h2: number[] = []
  const rhoMin: number[] = []
  const rhoMax: number[] = []

  let R = 0
  for (const v of domainVertices) R = Math.max(R, Math.hypot(v[0], v[1], v[2]))
  let maxR = 0
  for (let i = 0; i < n; i++)
    maxR = Math.max(
      maxR,
      cells.radius[i] + Math.hypot(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]),
    )
  const Bi = inv3([
    [basis[0][0], basis[1][0], basis[2][0]],
    [basis[0][1], basis[1][1], basis[2][1]],
    [basis[0][2], basis[1][2], basis[2][2]],
  ])
  const zMax = [0, 1, 2].map((k) => Math.ceil(Math.hypot(Bi[k][0], Bi[k][1], Bi[k][2]) * (R + maxR)))
  const bScale = Math.max(1, Math.abs(Math.max(...domainB)))

  let maxK = 0
  for (let i = 0; i < n; i++) maxK = Math.max(maxK, planeStart[i + 1] - planeStart[i])
  let ax = new Float64Array(64 + maxK)
  let ay = new Float64Array(64 + maxK)
  let bx = new Float64Array(64 + maxK)
  let by = new Float64Array(64 + maxK)

  for (let fi = 0; fi < H; fi++) {
    const a = domainA[fi]
    const an = Math.hypot(a[0], a[1], a[2])
    const nx = a[0] / an
    const ny = a[1] / an
    const nz = a[2] / an
    const dPlane = (3 * domainB[fi]) / an
    const onPlane = domainVertices.filter(
      (v) => Math.abs(a[0] * v[0] + a[1] * v[1] + a[2] * v[2] - 3 * domainB[fi]) < 1e-6 * bScale * an,
    )
    // in-plane orthonormal basis, right-handed with n: counterclockwise in
    // (e1, e2) is counterclockwise seen from outside
    const hx = Math.abs(nx) < 0.9 ? 1 : 0
    const hy = 1 - hx
    const hd = hx * nx + hy * ny
    let e1x = hx - hd * nx
    let e1y = hy - hd * ny
    let e1z = -hd * nz
    const e1n = Math.hypot(e1x, e1y, e1z)
    e1x /= e1n
    e1y /= e1n
    e1z /= e1n
    const e2x = ny * e1z - nz * e1y
    const e2y = nz * e1x - nx * e1z
    const e2z = nx * e1y - ny * e1x
    const ox = nx * dPlane
    const oy = ny * dPlane
    const oz = nz * dPlane
    frames.set([ox, oy, oz, e1x, e1y, e1z, e2x, e2y, e2z, nx, ny, nz], 12 * fi)
    if (onPlane.length < 3) continue

    // facet outline in the frame, counterclockwise
    const pts = onPlane.map((v) => {
      const dx = v[0] - ox
      const dy = v[1] - oy
      const dz = v[2] - oz
      return [dx * e1x + dy * e1y + dz * e1z, dx * e2x + dy * e2y + dz * e2z]
    })
    const gu = pts.reduce((s, p) => s + p[0], 0) / pts.length
    const gv = pts.reduce((s, p) => s + p[1], 0) / pts.length
    pts.sort((p, q) => Math.atan2(p[1] - gv, p[0] - gu) - Math.atan2(q[1] - gv, q[0] - gu))
    let facetR = 0
    for (const p of pts) facetR = Math.max(facetR, Math.hypot(p[0] - gu, p[1] - gv))

    for (let i = 0; i < n; i++) {
      const p0 = planeStart[i]
      const p1 = planeStart[i + 1]
      const Ri = cells.radius[i]
      if (p1 - p0 < 4 || Ri === 0) continue
      for (let z0 = -zMax[0]; z0 <= zMax[0]; z0++)
        for (let z1 = -zMax[1]; z1 <= zMax[1]; z1++)
          for (let z2 = -zMax[2]; z2 <= zMax[2]; z2++) {
            const sx = centers[3 * i] + z0 * basis[0][0] + z1 * basis[1][0] + z2 * basis[2][0]
            const sy = centers[3 * i + 1] + z0 * basis[0][1] + z1 * basis[1][1] + z2 * basis[2][1]
            const sz = centers[3 * i + 2] + z0 * basis[0][2] + z1 * basis[1][2] + z2 * basis[2][2]
            const dn = nx * sx + ny * sy + nz * sz - dPlane
            if (Math.abs(dn) > Ri) continue
            const su = (sx - ox) * e1x + (sy - oy) * e1y + (sz - oz) * e1z
            const sv = (sx - ox) * e2x + (sy - oy) * e2y + (sz - oz) * e2z
            if (Math.hypot(su - gu, sv - gv) > Ri + facetR) continue

            let m = pts.length
            for (let v = 0; v < m; v++) {
              ax[v] = pts[v][0]
              ay[v] = pts[v][1]
            }
            // cell plane n_k · (x - s) <= c_k with x = origin + u e1 + v e2
            for (let p = p0; p < p1 && m >= 3; p++) {
              const kx = planes[4 * p]
              const ky = planes[4 * p + 1]
              const kz = planes[4 * p + 2]
              const ca = kx * e1x + ky * e1y + kz * e1z
              const cb = kx * e2x + ky * e2y + kz * e2z
              const cc = planes[4 * p + 3] - (kx * (ox - sx) + ky * (oy - sy) + kz * (oz - sz))
              m = clip2(ax, ay, m, ca, cb, cc, bx, by)
              let tmp = ax
              ax = bx
              bx = tmp
              tmp = ay
              ay = by
              by = tmp
            }
            m = dedupe2(ax, ay, m, (1e-9 * facetR) ** 2)
            if (m < 3) continue
            let area = 0
            for (let v = 0; v < m; v++) {
              const w = v + 1 === m ? 0 : v + 1
              area += ax[v] * ay[w] - ax[w] * ay[v]
            }
            if (!(area > 1e-14 * facetR * facetR)) continue

            // nearest / farthest point of the piece from the projected site
            const [rmin, rmax] = pieceReach(ax, ay, m, su, sv)
            site.push(i)
            center.push(sx, sy, sz)
            facet.push(fi)
            for (let v = 0; v < m; v++) poly.push(ax[v], ay[v])
            polyStart.push(poly.length / 2)
            o.push(su, sv)
            h2.push(dn * dn)
            rhoMin.push(rmin)
            rhoMax.push(rmax)
          }
    }
  }

  return {
    count: site.length,
    site: Uint32Array.from(site),
    center: Float64Array.from(center),
    facet: Uint32Array.from(facet),
    frames,
    polyStart: Uint32Array.from(polyStart),
    poly: Float64Array.from(poly),
    o: Float64Array.from(o),
    h2: Float64Array.from(h2),
    rhoMin: Float64Array.from(rhoMin),
    rhoMax: Float64Array.from(rhoMax),
    wholeOwner: new Int32Array(site.length).fill(-2),
  }
}

const CIRCLE_SEGMENTS = 128

// Per piece, the piece ∩ disk (voronoi false) or the piece minus the disk
// (voronoi true), where the disk is the piece's site ball of squared radius
// level + w cut by the piece's plane; triangles in WORLD space through the
// facet frames. `ownerOf(piece, u, v)` names the component of a frame point
// outside the disk; without it triangles are owned by the piece's site.
function pieceMesh(
  caps: CapPieces,
  weights: Float64Array,
  level: number,
  voronoi: boolean,
  ownerOf: ((p: number, u: number, v: number) => number) | null,
): IsosurfaceMesh {
  const out = meshWriter(4096)
  const { frames, poly, polyStart } = caps
  const N = CIRCLE_SEGMENTS
  let maxM = 0
  for (let p = 0; p < caps.count; p++) maxM = Math.max(maxM, polyStart[p + 1] - polyStart[p])
  const size = N + 2 * maxM + 8
  let cx = new Float64Array(size)
  let cy = new Float64Array(size)
  let dx = new Float64Array(size)
  let dy = new Float64Array(size)
  const ang = new Float64Array(size)
  const qu = new Float64Array(maxM) // the current piece's outline
  const qv = new Float64Array(maxM)

  // one planar triangle of facet frame f, counterclockwise in (u, v)
  const emit = (
    f: number, u0: number, v0: number, u1: number, v1: number, u2: number, v2: number,
    owner: number,
  ) => {
    const b = 12 * f
    out.reserve(1)
    out.tri(
      frames[b] + u0 * frames[b + 3] + v0 * frames[b + 6],
      frames[b + 1] + u0 * frames[b + 4] + v0 * frames[b + 7],
      frames[b + 2] + u0 * frames[b + 5] + v0 * frames[b + 8],
      frames[b] + u1 * frames[b + 3] + v1 * frames[b + 6],
      frames[b + 1] + u1 * frames[b + 4] + v1 * frames[b + 7],
      frames[b + 2] + u1 * frames[b + 5] + v1 * frames[b + 8],
      frames[b] + u2 * frames[b + 3] + v2 * frames[b + 6],
      frames[b + 1] + u2 * frames[b + 4] + v2 * frames[b + 7],
      frames[b + 2] + u2 * frames[b + 5] + v2 * frames[b + 8],
      frames[b + 9], frames[b + 10], frames[b + 11],
      frames[b + 9], frames[b + 10], frames[b + 11],
      frames[b + 9], frames[b + 10], frames[b + 11],
      owner,
    )
  }
  // where the ray from (gu, gv) at angle th leaves the counterclockwise
  // convex polygon (xs, ys)[0..m) around it; writes hitU/hitV
  let hitU = 0
  let hitV = 0
  const hit = (xs: Float64Array, ys: Float64Array, m: number, gu: number, gv: number, th: number) => {
    const du = Math.cos(th)
    const dv = Math.sin(th)
    let t = Infinity
    for (let v = 0; v < m; v++) {
      const w = v + 1 === m ? 0 : v + 1
      const x0 = xs[v]
      const y0 = ys[v]
      const ex = xs[w] - x0
      const ey = ys[w] - y0
      const den = ex * dv - ey * du // rate at which the ray leaves this edge's halfplane
      if (den < 0) {
        const num = ex * (gv - y0) - ey * (gu - x0)
        const tt = num / -den
        if (tt < t) t = tt
      }
    }
    if (!Number.isFinite(t) || t < 0) t = 0
    hitU = gu + t * du
    hitV = gv + t * dv
  }

  for (let p = 0; p < caps.count; p++) {
    const i = caps.site[p]
    const f = caps.facet[p]
    const v0 = polyStart[p]
    const m = polyStart[p + 1] - v0
    for (let v = 0; v < m; v++) {
      qu[v] = poly[2 * (v0 + v)]
      qv[v] = poly[2 * (v0 + v) + 1]
    }
    const rho2 = level + weights[i] - caps.h2[p]
    const rho = rho2 > 0 ? Math.sqrt(rho2) : 0
    const disjoint = rho <= caps.rhoMin[p] // the disk misses the piece
    const covered = rho >= caps.rhoMax[p] // the disk covers the piece
    if (voronoi ? covered : disjoint) continue

    if (voronoi ? disjoint : covered) {
      let owner = i
      if (ownerOf) {
        if (caps.wholeOwner[p] === -2) {
          let gu = 0
          let gv = 0
          for (let v = 0; v < m; v++) {
            gu += qu[v]
            gv += qv[v]
          }
          caps.wholeOwner[p] = ownerOf(p, gu / m, gv / m)
        }
        owner = caps.wholeOwner[p]
      }
      for (let v = 1; v + 1 < m; v++)
        emit(f, qu[0], qv[0], qu[v], qv[v], qu[v + 1], qv[v + 1], owner)
      continue
    }

    // the disk outline as a polygon, clipped to the piece: C = piece ∩ disk
    const ou = caps.o[2 * p]
    const ov = caps.o[2 * p + 1]
    let k = N
    for (let s = 0; s < N; s++) {
      const th = (2 * Math.PI * s) / N
      cx[s] = ou + rho * Math.cos(th)
      cy[s] = ov + rho * Math.sin(th)
    }
    for (let v = 0; v < m && k >= 3; v++) {
      const w = v + 1 === m ? 0 : v + 1
      const x0 = qu[v]
      const y0 = qv[v]
      const ex = qu[w] - x0
      const ey = qv[w] - y0
      // inside the counterclockwise piece: ex (y - y0) - ey (x - x0) >= 0
      k = clip2(cx, cy, k, ey, -ex, ey * x0 - ex * y0, dx, dy)
      let tmp = cx
      cx = dx
      dx = tmp
      tmp = cy
      cy = dy
      dy = tmp
    }
    k = dedupe2(cx, cy, k, (1e-9 * caps.rhoMax[p]) ** 2)
    let areaC = 0
    for (let v = 0; v < k; v++) {
      const w = v + 1 === k ? 0 : v + 1
      areaC += cx[v] * cy[w] - cx[w] * cy[v]
    }
    const hasC = k >= 3 && areaC > 1e-14 * caps.rhoMax[p] * caps.rhoMax[p]

    if (!voronoi) {
      if (!hasC) continue
      for (let v = 1; v + 1 < k; v++) emit(f, cx[0], cy[0], cx[v], cy[v], cx[v + 1], cy[v + 1], i)
      continue
    }
    if (!hasC) {
      // the disk only grazes the piece: all of it is outside
      const owner = ownerOf ? ownerOf(p, qu[0], qv[0]) : i
      for (let v = 1; v + 1 < m; v++)
        emit(f, qu[0], qv[0], qu[v], qv[v], qu[v + 1], qv[v + 1], owner)
      continue
    }

    // piece minus C: sweep around a point of C; between consecutive vertex
    // directions of either polygon both outlines are straight, so the ring
    // sector between them is one exact quad
    let gu = 0
    let gv = 0
    for (let v = 0; v < k; v++) {
      gu += cx[v]
      gv += cy[v]
    }
    gu /= k
    gv /= k
    let na = 0
    for (let v = 0; v < k; v++) ang[na++] = Math.atan2(cy[v] - gv, cx[v] - gu)
    for (let v = 0; v < m; v++) ang[na++] = Math.atan2(qv[v] - gv, qu[v] - gu)
    const sorted = ang.subarray(0, na).sort()
    const thin = 1e-9 * caps.rhoMax[p]
    let runOwner = -1
    let inRun = false
    hit(cx, cy, k, gu, gv, sorted[na - 1])
    let ciu = hitU
    let civ = hitV
    hit(qu, qv, m, gu, gv, sorted[na - 1])
    let qiu = hitU
    let qiv = hitV
    for (let s = 0; s < na; s++) {
      const th = sorted[s]
      if (s > 0 && th - sorted[s - 1] < 1e-12) continue // coincident directions
      hit(cx, cy, k, gu, gv, th)
      const cju = hitU
      const cjv = hitV
      hit(qu, qv, m, gu, gv, th)
      const qju = hitU
      const qjv = hitV
      const wide =
        Math.hypot(qiu - ciu, qiv - civ) > thin || Math.hypot(qju - cju, qjv - cjv) > thin
      if (wide) {
        if (!inRun) {
          // a new connected stretch of the ring: one owner for all of it
          runOwner = ownerOf
            ? ownerOf(p, (ciu + qiu + qju + cju) / 4, (civ + qiv + qjv + cjv) / 4)
            : i
          inRun = true
        }
        emit(f, ciu, civ, qiu, qiv, qju, qjv, runOwner)
        emit(f, ciu, civ, qju, qjv, cju, cjv, runOwner)
      } else inRun = false
      ciu = cju
      civ = cjv
      qiu = qju
      qiv = qjv
    }
  }
  return out.finish()
}

// Caps closing the solid where the 3x domain boundary cuts it, in WORLD
// space (the domain boundary is not periodic): per piece, the piece ∩ disk
// (Delaunay ball union) or the piece minus the disk (Voronoi solid), where
// the disk is the site's ball cut by the facet plane. Same level, side and
// owner conventions as powerSurface.
export function powerCaps(
  cells: PowerCells,
  caps: CapPieces,
  level: number,
  voronoi: boolean,
  withOwners: boolean,
): IsosurfaceMesh {
  const { frames } = caps
  return pieceMesh(
    caps,
    cells.weights,
    level,
    voronoi,
    voronoi && withOwners
      ? (p, u, v) => {
          const b = 12 * caps.facet[p]
          return ascend(
            cells,
            caps.site[p],
            frames[b] + u * frames[b + 3] + v * frames[b + 6] - caps.center[3 * p],
            frames[b + 1] + u * frames[b + 4] + v * frames[b + 7] - caps.center[3 * p + 1],
            frames[b + 2] + u * frames[b + 5] + v * frames[b + 8] - caps.center[3 * p + 2],
          )
        }
      : null,
  )
}

// ---- 2D: exact filtration regions --------------------------------------------

export interface PowerCells2D {
  n: number
  centers: Float64Array // 2 per site
  weights: Float64Array
  // cell i = { y : n_k · y <= c_k } relative to its site; 3 numbers per line
  lineStart: Uint32Array // n + 1 offsets
  lines: Float64Array
  // cell vertices relative to the site, counterclockwise
  vertStart: Uint32Array // n + 1 offsets
  verts: Float64Array // 2 per cell vertex
  vertVor: Int32Array // quotient Voronoi vertex id per cell vertex, -1 unknown
  radius: Float64Array // per site: farthest cell vertex
  basis: number[][]
}

// The 2D power cells of the quotient sites (input as for buildPowerCells,
// with two coordinates; `level` is unused).
export function buildPowerCells2D(input: PowerCellsInput): PowerCells2D {
  const n = input.sites.length
  const centers = new Float64Array(2 * n)
  const weights = Float64Array.from(input.weights)
  for (let i = 0; i < n; i++) {
    centers[2 * i] = input.sites[i][0]
    centers[2 * i + 1] = input.sites[i][1]
  }
  const lists: number[][] = Array.from({ length: n }, () => [])
  let span = 0
  for (const a of input.arcs) {
    const dx = a.end[0] - a.start[0]
    const dy = a.end[1] - a.start[1]
    const d2 = dx * dx + dy * dy
    if (d2 === 0) continue
    span = Math.max(span, Math.sqrt(d2))
    const ws = weights[a.vStart]
    const we = weights[a.vEnd]
    lists[a.vStart].push(2 * dx, 2 * dy, d2 + ws - we)
    lists[a.vEnd].push(-2 * dx, -2 * dy, d2 + we - ws)
  }

  const lineStart = new Uint32Array(n + 1)
  const vertStart = new Uint32Array(n + 1)
  const radius = new Float64Array(n)
  const vertChunks: number[][] = []
  // a cell is cut out of a box far larger than any cell; one that still
  // touches the box is unbounded (degenerate input) and gets no vertices
  const box = 1e3 * (span || 1)
  let maxK = 0
  for (let i = 0; i < n; i++) maxK = Math.max(maxK, lists[i].length / 3)
  let ax = new Float64Array(maxK + 8)
  let ay = new Float64Array(maxK + 8)
  let bx = new Float64Array(maxK + 8)
  let by = new Float64Array(maxK + 8)
  for (let i = 0; i < n; i++) {
    const ln = lists[i]
    let m = 4
    ax.set([-box, box, box, -box])
    ay.set([-box, -box, box, box])
    for (let k = 0; k < ln.length && m >= 3; k += 3) {
      m = clip2(ax, ay, m, ln[k], ln[k + 1], ln[k + 2], bx, by)
      let tmp = ax
      ax = bx
      bx = tmp
      tmp = ay
      ay = by
      by = tmp
    }
    m = dedupe2(ax, ay, m, (1e-9 * (span || 1)) ** 2)
    const verts: number[] = []
    let r = 0
    let bounded = m >= 3
    for (let v = 0; v < m; v++) {
      if (Math.max(Math.abs(ax[v]), Math.abs(ay[v])) > 0.5 * box) bounded = false
      r = Math.max(r, Math.hypot(ax[v], ay[v]))
    }
    if (bounded) for (let v = 0; v < m; v++) verts.push(ax[v], ay[v])
    radius[i] = bounded ? r : 0
    vertChunks.push(verts)
    lineStart[i + 1] = lineStart[i] + ln.length / 3
    vertStart[i + 1] = vertStart[i] + verts.length / 2
  }
  const lines = Float64Array.from(lists.flat())
  const verts = Float64Array.from(vertChunks.flat())

  // name the cell vertices: nearest quotient Voronoi vertex modulo the lattice
  const vertVor = new Int32Array(verts.length / 2).fill(-1)
  const vor = input.vorVertices
  if (vor) {
    const B = input.basis
    // columns of M are the basis vectors: x = M z
    const det = B[0][0] * B[1][1] - B[1][0] * B[0][1]
    const tol2 = (1e-5 * Math.hypot(B[0][0], B[0][1])) ** 2
    for (let i = 0; i < n; i++) {
      for (let v = vertStart[i]; v < vertStart[i + 1]; v++) {
        const X = centers[2 * i] + verts[2 * v]
        const Y = centers[2 * i + 1] + verts[2 * v + 1]
        let best = -1
        let bestD = Infinity
        for (let id = 0; id < vor.length; id++) {
          const R = vor[id]
          if (!R) continue
          const dx = X - R[0]
          const dy = Y - R[1]
          let f0 = (B[1][1] * dx - B[1][0] * dy) / det
          let f1 = (B[0][0] * dy - B[0][1] * dx) / det
          f0 -= Math.round(f0)
          f1 -= Math.round(f1)
          const rx = B[0][0] * f0 + B[1][0] * f1
          const ry = B[0][1] * f0 + B[1][1] * f1
          const d2 = rx * rx + ry * ry
          if (d2 < bestD) {
            bestD = d2
            best = id
          }
        }
        if (bestD < tol2) vertVor[v] = best
      }
    }
  }

  return { n, centers, weights, lineStart, lines, vertStart, verts, vertVor, radius, basis: input.basis }
}

// 2D counterpart of ascend: from the point (x, y) of cell i (relative to the
// site) walk away from the site to a cell edge, then along it away from the
// site's foot to a vertex. The power distance to the site never decreases.
export function ascend2D(cells: PowerCells2D, i: number, x: number, y: number): number {
  const { lines, lineStart, verts, vertStart, vertVor } = cells
  const p0 = lineStart[i]
  const p1 = lineStart[i + 1]
  if (vertStart[i + 1] === vertStart[i]) return -1
  let j = -1
  let s = Infinity
  for (let p = p0; p < p1; p++) {
    const d = lines[3 * p] * x + lines[3 * p + 1] * y
    if (d > 0) {
      const sp = lines[3 * p + 2] / d
      if (sp < s) {
        s = sp
        j = p
      }
    }
  }
  let vx = x
  let vy = y
  if (j >= 0) {
    if (s < 1) s = 1
    vx = s * x
    vy = s * y
    const jx = lines[3 * j]
    const jy = lines[3 * j + 1]
    const jn2 = jx * jx + jy * jy
    const fs = lines[3 * j + 2] / jn2
    let dx = vx - fs * jx
    let dy = vy - fs * jy
    if (dx * dx + dy * dy < 1e-24 * jn2) {
      // the ray hit the foot itself: either way along the edge climbs
      dx = -jy
      dy = jx
    }
    const dn = Math.hypot(dx, dy)
    let u = Infinity
    for (let p = p0; p < p1; p++) {
      if (p === j) continue
      const px = lines[3 * p]
      const py = lines[3 * p + 1]
      const den = px * dx + py * dy
      if (den > 1e-12 * Math.hypot(px, py) * dn) {
        const up = (lines[3 * p + 2] - (px * vx + py * vy)) / den
        if (up < u) u = up
      }
    }
    if (Number.isFinite(u)) {
      if (u < 0) u = 0
      vx += u * dx
      vy += u * dy
    }
  }
  let best = -1
  let bestD = Infinity
  for (let v = vertStart[i]; v < vertStart[i + 1]; v++) {
    const dx = verts[2 * v] - vx
    const dy = verts[2 * v + 1] - vy
    const d2 = dx * dx + dy * dy
    if (d2 < bestD) {
      bestD = d2
      best = v
    }
  }
  return best < 0 ? -1 : vertVor[best]
}

// The 3x domain (its outline loop) cut into pieces by the power cells: the
// 2D analogue of buildCapPieces, with a single "facet" — the plane itself at
// height z, the drawing layer.
export function buildRegionPieces2D(
  cells: PowerCells2D,
  outline: number[][],
  z: number,
): CapPieces {
  const { n, centers, lines, lineStart, basis } = cells
  // counterclockwise outline
  let area2 = 0
  for (let v = 0; v < outline.length; v++) {
    const w = (v + 1) % outline.length
    area2 += outline[v][0] * outline[w][1] - outline[w][0] * outline[v][1]
  }
  const loop = area2 < 0 ? [...outline].reverse() : outline
  let R = 0
  for (const v of loop) R = Math.max(R, Math.hypot(v[0], v[1]))
  let maxR = 0
  let maxK = 0
  for (let i = 0; i < n; i++) {
    maxR = Math.max(maxR, cells.radius[i] + Math.hypot(centers[2 * i], centers[2 * i + 1]))
    maxK = Math.max(maxK, lineStart[i + 1] - lineStart[i])
  }
  // T = Σ z_k basis[k] ⇒ |z_k| <= |row k of the inverse| · |T|
  const det = basis[0][0] * basis[1][1] - basis[1][0] * basis[0][1]
  const zMax = [
    Math.ceil((Math.hypot(basis[1][1], basis[1][0]) / Math.abs(det)) * (R + maxR)),
    Math.ceil((Math.hypot(basis[0][1], basis[0][0]) / Math.abs(det)) * (R + maxR)),
  ]

  const site: number[] = []
  const center: number[] = []
  const polyStart: number[] = [0]
  const poly: number[] = []
  const o: number[] = []
  const rhoMin: number[] = []
  const rhoMax: number[] = []
  let ax = new Float64Array(loop.length + maxK + 8)
  let ay = new Float64Array(loop.length + maxK + 8)
  let bx = new Float64Array(loop.length + maxK + 8)
  let by = new Float64Array(loop.length + maxK + 8)
  for (let i = 0; i < n; i++) {
    const p0 = lineStart[i]
    const p1 = lineStart[i + 1]
    const Ri = cells.radius[i]
    if (Ri === 0) continue
    for (let z0 = -zMax[0]; z0 <= zMax[0]; z0++)
      for (let z1 = -zMax[1]; z1 <= zMax[1]; z1++) {
        const sx = centers[2 * i] + z0 * basis[0][0] + z1 * basis[1][0]
        const sy = centers[2 * i + 1] + z0 * basis[0][1] + z1 * basis[1][1]
        if (Math.hypot(sx, sy) > Ri + R) continue
        let m = loop.length
        for (let v = 0; v < m; v++) {
          ax[v] = loop[v][0]
          ay[v] = loop[v][1]
        }
        // cell line n_k · (x - s) <= c_k
        for (let p = p0; p < p1 && m >= 3; p++) {
          const kx = lines[3 * p]
          const ky = lines[3 * p + 1]
          m = clip2(ax, ay, m, kx, ky, lines[3 * p + 2] + kx * sx + ky * sy, bx, by)
          let tmp = ax
          ax = bx
          bx = tmp
          tmp = ay
          ay = by
          by = tmp
        }
        m = dedupe2(ax, ay, m, (1e-9 * R) ** 2)
        if (m < 3) continue
        let area = 0
        for (let v = 0; v < m; v++) {
          const w = v + 1 === m ? 0 : v + 1
          area += ax[v] * ay[w] - ax[w] * ay[v]
        }
        if (!(area > 1e-14 * R * R)) continue
        const [rmin, rmax] = pieceReach(ax, ay, m, sx, sy)
        site.push(i)
        center.push(sx, sy, z)
        for (let v = 0; v < m; v++) poly.push(ax[v], ay[v])
        polyStart.push(poly.length / 2)
        o.push(sx, sy)
        rhoMin.push(rmin)
        rhoMax.push(rmax)
      }
  }
  return {
    count: site.length,
    site: Uint32Array.from(site),
    center: Float64Array.from(center),
    facet: new Uint32Array(site.length),
    frames: Float64Array.from([0, 0, z, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    polyStart: Uint32Array.from(polyStart),
    poly: Float64Array.from(poly),
    o: Float64Array.from(o),
    h2: new Float64Array(site.length),
    rhoMin: Float64Array.from(rhoMin),
    rhoMax: Float64Array.from(rhoMax),
    wholeOwner: new Int32Array(site.length).fill(-2),
  }
}

// The exact 2D filtration region over the 3x domain as flat triangles:
// {pi <= level} (the disk union; voronoi false, triOwner = site) or
// {pi >= level} (the domain minus the disks; voronoi true, triOwner = a
// quotient Voronoi vertex of the component when `withOwners`).
export function powerRegion2D(
  cells: PowerCells2D,
  pieces: CapPieces,
  level: number,
  voronoi: boolean,
  withOwners: boolean,
): IsosurfaceMesh {
  return pieceMesh(
    pieces,
    cells.weights,
    level,
    voronoi,
    voronoi && withOwners
      ? (p, u, v) => ascend2D(cells, pieces.site[p], u - pieces.o[2 * p], v - pieces.o[2 * p + 1])
      : null,
  )
}
