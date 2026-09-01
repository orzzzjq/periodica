import ControlPanel from './components/ControlPanel'
import { BarcodePanel, DiagramPanel, ImagePanel, SharedRangeToggle } from './components/DescriptorPanel'
import PanelHost from './components/PanelSystem'
import RadiusSlider from './components/RadiusSlider'
import Scene from './components/Scene'
import './App.css'

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
}

const HEADER_EXTRAS = {
  image: <SharedRangeToggle />,
}

export default function App() {
  return <PanelHost contents={CONTENTS} headerExtras={HEADER_EXTRAS} />
}
