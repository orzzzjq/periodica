// The native "periodica geometry" text format (v1):
//
//   geometry:       magic header; the next line is the format version ("1")
//   dimension:      2 or 3
//   lattice:        d rows of the matrix U — the lattice vectors are its
//                   COLUMNS (matching the UI and the quotient files, NOT the
//                   POSCAR rows-are-vectors convention)
//   coordinates:    optional: "fractional" (default) | "real"
//   points:         a count line, then one point per row: d coordinates,
//                   optionally followed by a weight (uniform per file)
//
// '#' starts a comment, blank lines are ignored, tokens are separated by any
// whitespace. Sections appear in the fixed order above.

export interface GeometryFile {
  d: 2 | 3
  lattice: number[][]
  points: number[][]
  weights: number[]
  coordMode: 'fractional' | 'real'
}

interface Line {
  no: number
  tokens: string[]
}

function tokenize(text: string): Line[] {
  const out: Line[] = []
  text.split(/\r?\n/).forEach((raw, i) => {
    const s = raw.split('#', 1)[0].trim()
    if (s) out.push({ no: i + 1, tokens: s.split(/\s+/) })
  })
  return out
}

function det(m: number[][], d: number): number {
  return d === 2
    ? m[0][0] * m[1][1] - m[0][1] * m[1][0]
    : m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

export function parseGeometry(text: string): GeometryFile {
  const lines = tokenize(text)
  let pos = 0

  const fail = (no: number, msg: string): never => {
    throw new Error(`line ${no}: ${msg}`)
  }
  const take = (what: string): Line => {
    const l = lines[pos++]
    if (!l) throw new Error(`unexpected end of file (expected ${what})`)
    return l
  }
  const header = (name: string) => {
    const l = take(`'${name}'`)
    if (l.tokens.length !== 1 || l.tokens[0] !== name) fail(l.no, `expected '${name}'`)
  }
  const numbers = (l: Line, count: number, what: string): number[] => {
    if (l.tokens.length !== count)
      fail(l.no, `expected ${count} numbers (${what}), got ${l.tokens.length}`)
    return l.tokens.map((t) => {
      const v = Number(t)
      if (!Number.isFinite(v)) fail(l.no, `'${t}' is not a number`)
      return v
    })
  }

  if (!lines.length || lines[0].tokens[0] !== 'geometry:')
    throw new Error("not a periodica geometry file (first line must be 'geometry:')")
  header('geometry:')
  const ver = take('format version')
  if (ver.tokens.join(' ') !== '1')
    fail(ver.no, `unsupported geometry format version '${ver.tokens.join(' ')}' (expected 1)`)

  header('dimension:')
  const dLine = take('dimension value')
  const d = Number(dLine.tokens[0])
  if (dLine.tokens.length !== 1 || (d !== 2 && d !== 3)) fail(dLine.no, 'dimension must be 2 or 3')

  header('lattice:')
  const lattice: number[][] = []
  for (let i = 0; i < d; i++) lattice.push(numbers(take('lattice row'), d, `lattice row ${i + 1}`))
  if (Math.abs(det(lattice, d)) < 1e-12) throw new Error('lattice basis is singular')

  let coordMode: GeometryFile['coordMode'] = 'fractional'
  if (lines[pos]?.tokens.length === 1 && lines[pos].tokens[0] === 'coordinates:') {
    pos++
    const m = take('coordinate mode')
    const mode = m.tokens.join(' ')
    if (mode !== 'fractional' && mode !== 'real')
      fail(m.no, `coordinates must be 'fractional' or 'real'`)
    coordMode = mode as GeometryFile['coordMode']
  }

  header('points:')
  const nLine = take('point count')
  const n = Number(nLine.tokens[0])
  if (nLine.tokens.length !== 1 || !Number.isInteger(n) || n < 1)
    fail(nLine.no, 'point count must be a positive integer')
  const points: number[][] = []
  const weights: number[] = []
  let weighted: boolean | null = null
  for (let i = 0; i < n; i++) {
    const l = take(`point ${i + 1} of ${n}`)
    if (weighted === null) {
      if (l.tokens.length !== d && l.tokens.length !== d + 1)
        fail(l.no, `expected ${d} coordinates (optionally + weight), got ${l.tokens.length} numbers`)
      weighted = l.tokens.length === d + 1
    }
    const width = d + (weighted ? 1 : 0)
    const vals = numbers(l, width, weighted ? `${d} coordinates + weight` : `${d} coordinates`)
    points.push(vals.slice(0, d))
    weights.push(weighted ? vals[d] : 0)
  }
  if (lines[pos]) fail(lines[pos].no, 'unexpected content after the points')

  return { d: d as 2 | 3, lattice, points, weights, coordMode }
}

export function serializeGeometry(g: GeometryFile): string {
  // the weight column is written only when it carries information
  const weighted = g.weights.some((w) => w !== 0)
  const lines = [
    'geometry:',
    '1',
    'dimension:',
    String(g.d),
    'lattice:',
    '# matrix U printed row by row; the lattice vectors are its COLUMNS',
    ...g.lattice.map((r) => r.join(' ')),
    'coordinates:',
    g.coordMode,
    'points:',
    String(g.points.length),
    ...g.points.map((p, i) => (weighted ? [...p, g.weights[i] ?? 0] : p).join(' ')),
  ]
  return lines.join('\n') + '\n'
}
