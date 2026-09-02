export interface ComputeRequest {
  d: 2 | 3
  lattice: number[][]
  points: number[][] // n rows of d coordinates
  weights: number[]
  imageSize?: number
}

export interface Polytope2D {
  vertices: number[][]
  outline: number[][] // ordered loop
}

export interface Polytope3D {
  vertices: number[][]
  edges: number[][] // index pairs into vertices
  triangles: number[][] // index triples into vertices
}

export type Polytope = Polytope2D | Polytope3D

export interface QuotientArc {
  start: number[]
  end: number[]
  filtration: number
  shift: number[]
}

export interface Bar {
  birth: number
  death: number | null // null = infinite
  multiplicity: number
}

export interface ImagesData {
  size: number
  xmin: number
  xmax: number
  data: number[][][] // d+1 images
}

export interface Descriptors {
  barcodes: Bar[][] // d+1 lists
  images: ImagesData
}

export interface VoronoiGeometry {
  points3x: number[][] // full dual-skeleton vertices (circumcenters)
  fullEdges: number[][] // index pairs into points3x
  arcs: { start: number[]; end: number[]; filtration: number }[] // resolved periodic Voronoi edges
}

export interface ComputeResponse {
  d: 2 | 3
  voronoi: Descriptors | null
  voronoiError: string | null
  voronoiGeometry: VoronoiGeometry | null
  basis: number[][] // reduced basis vectors as rows
  domain1x: Polytope
  domain3x: Polytope
  domainA: number[][] // Dirichlet domain halfspaces: A·x <= b (1x), A·x <= 3b (3x)
  domainB: number[]
  points: {
    positions3x: number[][]
    originalIndex: number[]
    canonicalCount: number
    kept: number[]
    hidden: number[]
    weights: number[]
  }
  fullEdges: number[][] // index pairs into positions3x
  quotientArcs: QuotientArc[]
  maxRadius: number
  barcodes: Bar[][] // d+1 lists (Delaunay)
  images: ImagesData // (Delaunay)
}

/** True iff p satisfies A·x <= scale·b, i.e. lies in the scaled Dirichlet
 * domain (scale 1 = fundamental domain, 3 = the 3x construction domain). */
export function inDirichletDomain(
  p: number[],
  A: number[][],
  b: number[],
  scale = 3,
  eps = 1e-9,
): boolean {
  return A.every((row, i) => {
    const dot = row.reduce((s, a, j) => s + a * (p[j] ?? 0), 0)
    return dot <= scale * b[i] + eps
  })
}

export async function compute(req: ComputeRequest): Promise<ComputeResponse> {
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else detail = JSON.stringify(body.detail ?? body)
    } catch {
      /* keep default */
    }
    throw new Error(detail)
  }
  return res.json()
}
