// The native "periodica geometry" (v1) and "periodica grid" (v1) text
// formats. Both are sectioned: a magic header naming the format, a version
// line, then fixed-order sections. '#' starts a comment, blank lines are
// ignored, tokens are separated by any whitespace. The lattice block is the
// matrix U printed row by row — the lattice vectors are its COLUMNS
// (matching the UI and the quotient files, NOT the POSCAR rows-are-vectors
// convention).
//
// geometry:  dimension / lattice / coordinates (optional: "fractional"
//            (default) | "real") / points (count line, then one point per
//            row: d coordinates + optional weight, uniform per file)
// grid:      dimension / lattice / shape (N1..Nd) / values (one row of the
//            last axis per line, C order — first index slowest); the value
//            at [i1,...,id] belongs to the fractional point (i1/N1,...)

export interface GeometryFile {
  d: 2 | 3
  lattice: number[][]
  points: number[][]
  weights: number[]
  coordMode: 'fractional' | 'real'
}

export type GridValues = number[][] | number[][][]

export interface GridFile {
  d: 2 | 3
  lattice: number[][]
  values: GridValues
}

export type ParsedInput = ({ kind: 'geometry' } & GeometryFile) | ({ kind: 'grid' } & GridFile)

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

class SectionParser {
  lines: Line[]
  pos = 0

  constructor(text: string) {
    this.lines = tokenize(text)
  }

  fail(no: number, msg: string): never {
    throw new Error(`line ${no}: ${msg}`)
  }

  take(what: string): Line {
    const l = this.lines[this.pos++]
    if (!l) throw new Error(`unexpected end of file (expected ${what})`)
    return l
  }

  header(name: string) {
    const l = this.take(`'${name}'`)
    if (l.tokens.length !== 1 || l.tokens[0] !== name) this.fail(l.no, `expected '${name}'`)
  }

  peekHeader(name: string): boolean {
    const l = this.lines[this.pos]
    return !!l && l.tokens.length === 1 && l.tokens[0] === name
  }

  numbers(l: Line, count: number, what: string): number[] {
    if (l.tokens.length !== count)
      this.fail(l.no, `expected ${count} numbers (${what}), got ${l.tokens.length}`)
    return l.tokens.map((t) => {
      const v = Number(t)
      if (!Number.isFinite(v)) this.fail(l.no, `'${t}' is not a number`)
      return v
    })
  }

  magic(name: string, kind: string) {
    if (!this.lines.length || this.lines[0].tokens[0] !== name)
      throw new Error(`not a periodica ${kind} file (first line must be '${name}')`)
    this.header(name)
    const ver = this.take('format version')
    if (ver.tokens.join(' ') !== '1')
      this.fail(ver.no, `unsupported ${kind} format version '${ver.tokens.join(' ')}' (expected 1)`)
  }

  dimension(): 2 | 3 {
    this.header('dimension:')
    const l = this.take('dimension value')
    const d = Number(l.tokens[0])
    if (l.tokens.length !== 1 || (d !== 2 && d !== 3)) this.fail(l.no, 'dimension must be 2 or 3')
    return d as 2 | 3
  }

  lattice(d: number): number[][] {
    this.header('lattice:')
    const lattice: number[][] = []
    for (let i = 0; i < d; i++)
      lattice.push(this.numbers(this.take('lattice row'), d, `lattice row ${i + 1}`))
    if (Math.abs(det(lattice, d)) < 1e-12) throw new Error('lattice basis is singular')
    return lattice
  }

  done(what: string) {
    const l = this.lines[this.pos]
    if (l) this.fail(l.no, `unexpected content after the ${what}`)
  }
}

export function parseGeometry(text: string): GeometryFile {
  const p = new SectionParser(text)
  p.magic('geometry:', 'geometry')
  const d = p.dimension()
  const lattice = p.lattice(d)

  let coordMode: GeometryFile['coordMode'] = 'fractional'
  if (p.peekHeader('coordinates:')) {
    p.pos++
    const m = p.take('coordinate mode')
    const mode = m.tokens.join(' ')
    if (mode !== 'fractional' && mode !== 'real')
      p.fail(m.no, `coordinates must be 'fractional' or 'real'`)
    coordMode = mode as GeometryFile['coordMode']
  }

  p.header('points:')
  const nLine = p.take('point count')
  const n = Number(nLine.tokens[0])
  if (nLine.tokens.length !== 1 || !Number.isInteger(n) || n < 1)
    p.fail(nLine.no, 'point count must be a positive integer')
  const points: number[][] = []
  const weights: number[] = []
  let weighted: boolean | null = null
  for (let i = 0; i < n; i++) {
    const l = p.take(`point ${i + 1} of ${n}`)
    if (weighted === null) {
      if (l.tokens.length !== d && l.tokens.length !== d + 1)
        p.fail(l.no, `expected ${d} coordinates (optionally + weight), got ${l.tokens.length} numbers`)
      weighted = l.tokens.length === d + 1
    }
    const width = d + (weighted ? 1 : 0)
    const vals = p.numbers(l, width, weighted ? `${d} coordinates + weight` : `${d} coordinates`)
    points.push(vals.slice(0, d))
    weights.push(weighted ? vals[d] : 0)
  }
  p.done('points')

  return { d, lattice, points, weights, coordMode }
}

export function parseGrid(text: string): GridFile {
  const p = new SectionParser(text)
  p.magic('grid:', 'grid')
  const d = p.dimension()
  const lattice = p.lattice(d)

  p.header('shape:')
  const sLine = p.take('grid shape')
  const N = sLine.tokens.map(Number)
  if (N.length !== d || !N.every((n) => Number.isInteger(n) && n >= 1))
    p.fail(sLine.no, `shape must be ${d} positive integers`)

  p.header('values:')
  const last = N[N.length - 1]
  const nRows = N.slice(0, -1).reduce((a, b) => a * b, 1)
  const rows: number[][] = []
  for (let i = 0; i < nRows; i++)
    rows.push(p.numbers(p.take(`value row ${i + 1} of ${nRows}`), last, `${last} values`))
  p.done('values')

  const values: GridValues =
    d === 2 ? rows : Array.from({ length: N[0] }, (_, i) => rows.slice(i * N[1], (i + 1) * N[1]))
  return { d, lattice, values }
}

/** Dispatch a loaded input file on its magic header. */
export function parseInputFile(text: string): ParsedInput {
  const first = tokenize(text)[0]?.tokens[0]
  if (first === 'grid:') return { kind: 'grid', ...parseGrid(text) }
  // unknown headers fall through to the geometry error message
  return { kind: 'geometry', ...parseGeometry(text) }
}

export function gridShape(values: GridValues): number[] {
  return Array.isArray(values[0][0])
    ? [values.length, (values as number[][][])[0].length, (values as number[][][])[0][0].length]
    : [values.length, (values as number[][])[0].length]
}

const LATTICE_NOTE = '# matrix U printed row by row; the lattice vectors are its COLUMNS'

export function serializeGeometry(g: GeometryFile): string {
  // the weight column is written only when it carries information
  const weighted = g.weights.some((w) => w !== 0)
  const lines = [
    'geometry:',
    '1',
    'dimension:',
    String(g.d),
    'lattice:',
    LATTICE_NOTE,
    ...g.lattice.map((r) => r.join(' ')),
    'coordinates:',
    g.coordMode,
    'points:',
    String(g.points.length),
    ...g.points.map((p, i) => (weighted ? [...p, g.weights[i] ?? 0] : p).join(' ')),
  ]
  return lines.join('\n') + '\n'
}

export function serializeGrid(g: { lattice: number[][]; values: GridValues }): string {
  const shape = gridShape(g.values)
  const rows: number[][] =
    shape.length === 2 ? (g.values as number[][]) : (g.values as number[][][]).flat()
  const lines = [
    'grid:',
    '1',
    'dimension:',
    String(shape.length),
    'lattice:',
    LATTICE_NOTE,
    ...g.lattice.map((r) => r.join(' ')),
    'shape:',
    shape.join(' '),
    'values:',
    ...rows.map((r) => r.join(' ')),
  ]
  return lines.join('\n') + '\n'
}
