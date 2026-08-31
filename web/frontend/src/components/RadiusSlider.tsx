import { useStore } from '../store'

export default function RadiusSlider() {
  const results = useStore((s) => s.results)
  const radius = useStore((s) => s.ui.radius)
  const setUi = useStore((s) => s.setUi)
  const max = results?.maxRadius ?? 1

  return (
    <div className="slider-row">
      <span>R = {radius.toFixed(3)}</span>
      <input
        type="range"
        min={0}
        max={max}
        step={max / 400}
        value={Math.min(radius, max)}
        onChange={(e) => setUi({ radius: e.target.valueAsNumber })}
      />
    </div>
  )
}
