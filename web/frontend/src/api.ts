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

export interface ComputeResponse {
  d: 2 | 3
  basis: number[][] // reduced basis vectors as rows
  domain1x: Polytope
  domain3x: Polytope
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
  barcodes: Bar[][] // d+1 lists
  images: {
    size: number
    xmin: number
    xmax: number
    data: number[][][] // d+1 images
  }
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
