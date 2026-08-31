import ControlPanel from './components/ControlPanel'
import { BarcodePanel, DiagramPanel, ImagePanel } from './components/DescriptorPanel'
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

export default function App() {
  return <PanelHost contents={CONTENTS} />
}
