import ControlPanel from './components/ControlPanel'
import {
  BarcodePanel,
  DiagramPanel,
  ImagePanel,
  MergeTreePanel,
  SharedRangeToggle,
  TreeDisplayOptions,
} from './components/DescriptorPanel'
import PanelHost from './components/PanelSystem'
import RadiusSlider from './components/RadiusSlider'
import Scene, { DisplayOptions } from './components/Scene'
import { useStore } from './store'
import './App.css'

// Mutually exclusive complex selector for the descriptor panels.
function ComplexToggle() {
  const complexType = useStore((s) => s.ui.complexType)
  const setUi = useStore((s) => s.setUi)
  return (
    <div className="complex-toggle">
      {(['delaunay', 'voronoi'] as const).map((t) => (
        <button
          key={t}
          className={`chip${complexType === t ? ' active' : ''}`}
          onClick={() => setUi({ complexType: t })}
        >
          {t === 'delaunay' ? 'Delaunay' : 'Voronoi'}
        </button>
      ))}
    </div>
  )
}

const CONTENTS = {
  input: <ControlPanel />,
  scene: (
    <div className="scene-panel">
      <div className="scene">
        <Scene />
      </div>
      <RadiusSlider />
    </div>
  ),
  barcode: <BarcodePanel />,
  diagram: <DiagramPanel />,
  image: <ImagePanel />,
  tree: <MergeTreePanel />,
}

const HEADER_EXTRAS = {
  scene: <DisplayOptions />,
  image: <SharedRangeToggle />,
  tree: <TreeDisplayOptions />,
}

export default function App() {
  return <PanelHost contents={CONTENTS} headerExtras={HEADER_EXTRAS} appBarExtra={<ComplexToggle />} />
}
