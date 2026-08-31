// Visual check: screenshot every preset, the descriptor tabs, and a slider drag.
// Usage: node scripts/visual-check.mjs [outDir]
import { chromium } from 'playwright'

const OUT = process.argv[2] ?? '/tmp/pshots'
const BASE = process.env.PERIODICA_URL ?? 'http://localhost:8000'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1680, height: 940 } })

const problems = []
page.on('console', (m) => {
  if (m.type() === 'error') problems.push(`console.error: ${m.text()}`)
})
page.on('pageerror', (e) => problems.push(`pageerror: ${e}`))
page.on('requestfailed', (r) => problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`))

const settle = async (ms = 1400) => page.waitForTimeout(ms)
const shot = async (name) => {
  await page.screenshot({ path: `${OUT}/${name}.png` })
  console.log(`shot: ${name}`)
}

await page.goto(BASE, { waitUntil: 'networkidle' })
await settle(2000)
await shot('00-initial')

const presets = [
  '2D square',
  '2D hexagonal',
  '2D weighted',
  '2D hidden point',
  '3D cubic',
  '3D two points',
]
for (const name of presets) {
  await page.selectOption('select', { label: name })
  await settle()
  await shot(name.replaceAll(' ', '-'))
}

// slider on the hidden-point preset (2D, weighted): balls should grow
await page.selectOption('select', { label: '2D hidden point' })
await settle()
await page.locator('input[type="range"]').evaluate((el, v) => {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
  setter.call(el, v)
  el.dispatchEvent(new Event('input', { bubbles: true }))
}, String(0.35))
await settle(600)
await shot('slider-r035')

// descriptor tabs on the 2D weighted preset
await page.selectOption('select', { label: '2D weighted' })
await settle()
for (const tab of ['Diagram', 'Image']) {
  await page.getByRole('button', { name: tab, exact: true }).click()
  await settle(900)
  await shot(`tab-${tab}`)
}

console.log(problems.length ? `PROBLEMS:\n${problems.join('\n')}` : 'no console/page/request errors')
await browser.close()
