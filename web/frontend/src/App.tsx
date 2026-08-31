import ControlPanel from './components/ControlPanel'
import DescriptorPanel from './components/DescriptorPanel'
import RadiusSlider from './components/RadiusSlider'
import Scene from './components/Scene'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <ControlPanel />
      <div className="scene-column">
        <div className="scene">
          <Scene />
        </div>
        <RadiusSlider />
      </div>
      <DescriptorPanel />
    </div>
  )
}
