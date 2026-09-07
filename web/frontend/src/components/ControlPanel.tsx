import { useEffect, useState } from 'react'
import { PRESETS } from '../presets'
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
  const applyPreset = useStore((s) => s.applyPreset)
  const applyRandom = useStore((s) => s.applyRandom)
  const setDimension = useStore((s) => s.setDimension)
  const setCoordMode = useStore((s) => s.setCoordMode)
  const computeNow = useStore((s) => s.computeNow)
  const dirty = useStore((s) => s.dirty)

  const { d, lattice, points, weights, coordMode } = inputs

  // Random preset config; null = a regular preset (or manual input) is active.
  // Any change of seed / point count / dimension regenerates deterministically.
  const [randomCfg, setRandomCfg] = useState<{ seed: number; n: number } | null>(null)
  useEffect(() => {
    if (randomCfg) applyRandom(randomCfg.seed, randomCfg.n)
  }, [randomCfg, d, applyRandom])
  const coordLabels = ['x', 'y', 'z'].slice(0, d)

  return (
    <div className="panel">
      <section>
        <label className="row">
          Preset{' '}
          <select
            defaultValue=""
            onChange={(e) => {
              if (e.target.value === '__random') {
                setRandomCfg((cfg) => cfg ?? { seed: 1, n: 2 })
                return
              }
              setRandomCfg(null)
              const p = PRESETS.find((p) => p.name === e.target.value)
              if (p) applyPreset(p)
            }}
          >
            <option value="" disabled>
              choose…
            </option>
            {PRESETS.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
            <option value="__random">Random</option>
          </select>
        </label>
        <div className="row">
          Dimension{' '}
          <button className={d === 2 ? 'active' : ''} onClick={() => setDimension(2)}>2D</button>
          <button className={d === 3 ? 'active' : ''} onClick={() => setDimension(3)}>3D</button>
        </div>
      </section>

      <section>
        {randomCfg && (
          <div className="row">
            seed{' '}
            <Num
              value={randomCfg.seed}
              step={1}
              style={{ width: 70 }}
              onChange={(v) => setRandomCfg({ ...randomCfg, seed: Math.round(v) })}
            />
            points{' '}
            <Num
              value={randomCfg.n}
              step={1}
              min={1}
              style={{ width: 70 }}
              onChange={(v) => setRandomCfg({ ...randomCfg, n: Math.max(1, Math.round(v)) })}
            />
          </div>
        )}
        <div className="row">
          <button
            className={dirty ? 'active' : ''}
            style={{ width: '100%' }}
            onClick={computeNow}
            title="run the pipeline with the current inputs"
          >
            Compute
          </button>
        </div>
        <h2>Lattice basis (columns = vectors)</h2>
        <div className="matrix" style={{ gridTemplateColumns: `repeat(${d}, 1fr)` }}>
          {lattice.map((row, i) =>
            row.map((v, j) => (
              <Num key={`${i}-${j}`} value={v} onChange={(x) => setLatticeEntry(i, j, x)} />
            )),
          )}
        </div>
      </section>

      <section>
        <h2>Points &amp; weights</h2>
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
      </section>

      <section className="status">
        {status === 'loading' && <div className="loading">computing…</div>}
        {error && <div className="error">{error}</div>}
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
