import { PRESETS } from '../presets'
import { useStore } from '../store'

function Num({ value, onChange, step = 0.05, min }: {
  value: number
  onChange: (v: number) => void
  step?: number
  min?: number
}) {
  return (
    <input
      type="number"
      className="num"
      value={Number.isFinite(value) ? value : ''}
      step={step}
      min={min}
      onChange={(e) => {
        const v = e.target.valueAsNumber
        if (!Number.isNaN(v)) onChange(v)
      }}
    />
  )
}

export default function ControlPanel() {
  const inputs = useStore((s) => s.inputs)
  const results = useStore((s) => s.results)
  const status = useStore((s) => s.status)
  const error = useStore((s) => s.error)
  const ui = useStore((s) => s.ui)
  const setUi = useStore((s) => s.setUi)
  const setLatticeEntry = useStore((s) => s.setLatticeEntry)
  const setPointCoord = useStore((s) => s.setPointCoord)
  const setWeight = useStore((s) => s.setWeight)
  const addPoint = useStore((s) => s.addPoint)
  const removePoint = useStore((s) => s.removePoint)
  const applyPreset = useStore((s) => s.applyPreset)
  const setDimension = useStore((s) => s.setDimension)

  const { d, lattice, points, weights } = inputs
  const coordLabels = ['x', 'y', 'z'].slice(0, d)

  return (
    <div className="panel">
      <section>
        <label className="row">
          Preset{' '}
          <select
            defaultValue=""
            onChange={(e) => {
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
          </select>
        </label>
        <div className="row">
          Dimension{' '}
          <button className={d === 2 ? 'active' : ''} onClick={() => setDimension(2)}>2D</button>
          <button className={d === 3 ? 'active' : ''} onClick={() => setDimension(3)}>3D</button>
        </div>
      </section>

      <section>
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

      <section>
        <h2>Display</h2>
        <label className="row">
          <input
            type="checkbox"
            checked={ui.showFullSkeleton}
            onChange={(e) => setUi({ showFullSkeleton: e.target.checked })}
          />
          full Delaunay skeleton
        </label>
        <label className="row">
          <input type="checkbox" checked={ui.showDomains} onChange={(e) => setUi({ showDomains: e.target.checked })} />
          Dirichlet domains
        </label>
        <label className="row">
          <input type="checkbox" checked={ui.showBalls} onChange={(e) => setUi({ showBalls: e.target.checked })} />
          filtration balls
        </label>
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
