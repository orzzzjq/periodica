import { useEffect, useRef, useState } from 'react'
import { saveFile } from '../capture'
import { parseGeometry, serializeGeometry } from '../geometry'
import { useStore } from '../store'

function Num({ value, onChange, step = 0.05, min, style }: {
  value: number
  onChange: (v: number) => void
  step?: number
  min?: number
  style?: React.CSSProperties
}) {
  // While focused, the field shows the raw draft so intermediate states
  // ("", "-", "0.") survive typing; a fully cleared field commits 0. On
  // blur the draft is dropped and the committed value is shown again.
  const [draft, setDraft] = useState<string | null>(null)
  return (
    <input
      type="number"
      className="num"
      style={style}
      value={draft ?? (Number.isFinite(value) ? value : '')}
      step={step}
      min={min}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = e.target.valueAsNumber
        if (!Number.isNaN(v)) onChange(v)
        // empty and NOT a half-typed number ("-", "1e"): treat as 0
        else if (!e.target.validity.badInput) onChange(0)
      }}
      onBlur={() => setDraft(null)}
    />
  )
}

export default function ControlPanel() {
  const inputs = useStore((s) => s.inputs)
  const results = useStore((s) => s.results)
  const status = useStore((s) => s.status)
  const error = useStore((s) => s.error)
  const setLatticeEntry = useStore((s) => s.setLatticeEntry)
  const setPointCoord = useStore((s) => s.setPointCoord)
  const setWeight = useStore((s) => s.setWeight)
  const addPoint = useStore((s) => s.addPoint)
  const removePoint = useStore((s) => s.removePoint)
  const applyRandom = useStore((s) => s.applyRandom)
  const loadGeometry = useStore((s) => s.loadGeometry)
  const hasGeometry = useStore((s) => s.hasGeometry)
  const setDimension = useStore((s) => s.setDimension)
  const setCoordMode = useStore((s) => s.setCoordMode)
  const computeNow = useStore((s) => s.computeNow)
  const dirty = useStore((s) => s.dirty)

  const { d, lattice, points, weights, coordMode } = inputs

  // Random-input config; null = a loaded file (or manual input) is active.
  // Any change of seed / point count / dimension regenerates deterministically.
  const [randomCfg, setRandomCfg] = useState<{ seed: number; n: number } | null>(null)
  useEffect(() => {
    if (randomCfg) applyRandom(randomCfg.seed, randomCfg.n)
  }, [randomCfg, d, applyRandom])
  const coordLabels = ['x', 'y', 'z'].slice(0, d)

  // section collapse; by default the lattice is shown and the points are not
  const [showLattice, setShowLattice] = useState(true)
  const [showPoints, setShowPoints] = useState(false)

  // geometry-file load/save; parse/save errors show under the Compute button
  const fileInput = useRef<HTMLInputElement>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const loadFile = async (file: File) => {
    try {
      const g = parseGeometry(await file.text())
      setRandomCfg(null) // a live Random preset would regenerate over the loaded data
      loadGeometry(g)
      setFileError(null)
    } catch (e) {
      setFileError(e instanceof Error ? e.message : String(e))
    }
  }
  const saveGeometryFile = async () => {
    try {
      const blob = new Blob([serializeGeometry(inputs)], { type: 'text/plain' })
      await saveFile('geometry.txt', blob)
    } catch (e) {
      setFileError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="panel">
      <section>
        <div className="row">
          File{' '}
          <button onClick={() => fileInput.current?.click()} title="load a periodica geometry file">
            Load
          </button>
          <button
            onClick={saveGeometryFile}
            disabled={!hasGeometry}
            title="save the current input as a geometry file"
          >
            Save
          </button>
          <button
            className={randomCfg ? 'active' : ''}
            onClick={() => setRandomCfg((cfg) => cfg ?? { seed: 0, n: 2 })}
            title="generate a reproducible random input (seeded)"
          >
            Random
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".txt,text/plain"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) loadFile(f)
              e.target.value = '' // so the same file can be loaded again
            }}
          />
        </div>
        <div className="row">
          Dimension{' '}
          <button className={d === 2 ? 'active' : ''} onClick={() => setDimension(2)}>2D</button>
          <button className={d === 3 ? 'active' : ''} onClick={() => setDimension(3)}>3D</button>
        </div>
        {randomCfg && (
          <div className="row">
            Seed{' '}
            <Num
              value={randomCfg.seed}
              step={1}
              style={{ width: 70 }}
              onChange={(v) => setRandomCfg({ ...randomCfg, seed: Math.round(v) })}
            />
            Points{' '}
            <Num
              value={randomCfg.n}
              step={1}
              min={1}
              style={{ width: 70 }}
              onChange={(v) => setRandomCfg({ ...randomCfg, n: Math.max(1, Math.round(v)) })}
            />
          </div>
        )}
      </section>

      <section>
        <div className="row">
          <button
            className={dirty && hasGeometry ? 'active' : ''}
            style={{ width: '100%' }}
            disabled={!hasGeometry}
            onClick={() => {
              setFileError(null)
              computeNow()
            }}
            title="run the pipeline with the current inputs"
          >
            Compute
          </button>
        </div>
        <div className="status">
          {status === 'loading' && <div className="loading">computing…</div>}
          {fileError && <div className="error">{fileError}</div>}
          {error && <div className="error">{error}</div>}
        </div>
      </section>

      {hasGeometry && (
        <section>
          <h2
            className="collapsible"
            onClick={() => setShowLattice((v) => !v)}
            title="click to expand/collapse"
          >
            {showLattice ? '▾' : '▸'} Lattice basis (columns = vectors)
          </h2>
          {showLattice && (
            <div className="matrix" style={{ gridTemplateColumns: `repeat(${d}, 1fr)` }}>
              {lattice.map((row, i) =>
                row.map((v, j) => (
                  <Num key={`${i}-${j}`} value={v} onChange={(x) => setLatticeEntry(i, j, x)} />
                )),
              )}
            </div>
          )}
        </section>
      )}

      {hasGeometry && (
        <section>
          <h2
            className="collapsible"
            onClick={() => setShowPoints((v) => !v)}
            title="click to expand/collapse"
          >
            {showPoints ? '▾' : '▸'} Points &amp; weights
          </h2>
          {showPoints && (
            <>
              <div className="row">
                <button
                  className={coordMode === 'fractional' ? 'active' : ''}
                  onClick={() => setCoordMode('fractional')}
                  title="coordinates are coefficients of the lattice basis vectors"
                >
                  fractional
                </button>
                <button
                  className={coordMode === 'real' ? 'active' : ''}
                  onClick={() => setCoordMode('real')}
                  title="coordinates are Cartesian, used as-is"
                >
                  real
                </button>
              </div>
              <table className="points">
                <thead>
                  <tr>
                    {coordLabels.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                    <th>w</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {points.map((p, row) => (
                    <tr key={row} className={results?.points.hidden.includes(row) ? 'hidden-point' : ''}>
                      {p.map((v, j) => (
                        <td key={j}>
                          <Num value={v} onChange={(x) => setPointCoord(row, j, x)} />
                        </td>
                      ))}
                      <td>
                        <Num value={weights[row]} onChange={(x) => setWeight(row, x)} step={0.01} min={0} />
                      </td>
                      <td>
                        <button onClick={() => removePoint(row)} disabled={points.length <= 1} title="remove">
                          ×
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button onClick={addPoint}>+ add point</button>
            </>
          )}
        </section>
      )}

      <section className="status">
        {results && results.points.hidden.length > 0 && (
          <div className="warning">
            hidden point{results.points.hidden.length > 1 ? 's' : ''} (dominated by weights):{' '}
            {results.points.hidden.join(', ')}
          </div>
        )}
      </section>
    </div>
  )
}
