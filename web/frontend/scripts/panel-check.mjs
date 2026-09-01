// Interaction check for the floating panel system.
// Usage: node scripts/panel-check.mjs [outDir]
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

const groupCount = () => page.locator('.panel-window').count()
const rectOf = async (sel) => page.locator(sel).boundingBox()
const windowWithTab = (tab) => `.panel-window[data-tabs*="${tab}"]`

const drag = async (fromX, fromY, toX, toY) => {
  await page.mouse.move(fromX, fromY)
  await page.mouse.down()
  for (let i = 1; i <= 10; i++) {
    await page.mouse.move(fromX + ((toX - fromX) * i) / 10, fromY + ((toY - fromY) * i) / 10)
  }
  await page.mouse.up()
  await page.waitForTimeout(250)
}

await page.goto(BASE, { waitUntil: 'networkidle' })
await page.evaluate(() => localStorage.removeItem('periodica-panels-v1'))
await page.reload({ waitUntil: 'networkidle' })
await page.waitForTimeout(2200)
await page.screenshot({ path: `${OUT}/panel-default.png` })
console.log('groups initially:', await groupCount())
if ((await groupCount()) !== 5) problems.push(`expected 5 initial groups, got ${await groupCount()}`)

// 1. drag the input window by its header
const input0 = await rectOf(windowWithTab('input'))
await drag(input0.x + 120, input0.y + 14, input0.x + 420, input0.y + 90)
const input1 = await rectOf(windowWithTab('input'))
if (Math.abs(input1.x - input0.x - 300) > 20 || Math.abs(input1.y - input0.y - 76) > 20)
  problems.push(`input window did not move as expected: ${JSON.stringify({ input0, input1 })}`)
else console.log('drag window: OK')

// 2. minimize the scene window into the app bar, restore via its chip
await page.locator(windowWithTab('scene')).locator('.win-btn').click()
await page.waitForTimeout(200)
const sceneHidden = (await page.locator(windowWithTab('scene')).count()) === 0
const chip = page.locator('.app-bar .chip', { hasText: 'Visualization' })
if (!sceneHidden || (await chip.count()) !== 1) problems.push('scene not minimized into app bar')
else console.log('minimize to app bar: OK')
await page.screenshot({ path: `${OUT}/panel-minimized.png` })
await chip.click()
await page.waitForTimeout(300)
const restored = (await page.locator(windowWithTab('scene') + ' canvas').count()) > 0
if (!restored || (await chip.count()) !== 0) problems.push('scene did not restore from chip')
else console.log('restore from chip: OK')

const dumpGroups = async (label) => {
  const tabs = await page.locator('.panel-window').evaluateAll((els) =>
    els.map((el) => el.getAttribute('data-tabs')),
  )
  console.log(`${label}:`, JSON.stringify(tabs))
}

// 3. drag the diagram window onto the barcode header -> merge into tabs
const diag0 = await rectOf('.panel-window[data-tabs="diagram"]')
const barcode0 = await rectOf(windowWithTab('barcode'))
await drag(diag0.x + 100, diag0.y + 14, barcode0.x + barcode0.width - 80, barcode0.y + 14)
await dumpGroups('after merge drag')
if ((await groupCount()) !== 4) problems.push(`merge failed: ${await groupCount()} groups`)
else console.log('merge by dragging onto header: OK')
await page.screenshot({ path: `${OUT}/panel-merged.png` })

// 4. drag the diagram tab back out into empty space (below the windows)
const diagramTab = page.locator('.tab', { hasText: 'Diagram' })
const tabBox = await diagramTab.boundingBox()
await drag(tabBox.x + tabBox.width / 2, tabBox.y + tabBox.height / 2, 300, 800)
await dumpGroups('after detach drag')
if ((await groupCount()) !== 5) problems.push(`detach failed: ${await groupCount()} groups`)
else console.log('detach tab -> floating window: OK')
await page.screenshot({ path: `${OUT}/panel-detached.png` })
await page.screenshot({ path: `${OUT}/panel-merged.png` })

// 5. persistence across reload
const before = await rectOf(windowWithTab('input'))
await page.reload({ waitUntil: 'networkidle' })
await page.waitForTimeout(2000)
const after = await rectOf(windowWithTab('input'))
if (Math.abs(before.x - after.x) > 2 || Math.abs(before.y - after.y) > 2)
  problems.push(`layout not persisted: ${JSON.stringify({ before, after })}`)
else console.log('persistence: OK')

// 6. scene still renders after all the shuffling (WebGL canvas alive)
const hasCanvas = await page.locator(windowWithTab('scene') + ' canvas').count()
if (!hasCanvas) problems.push('scene canvas missing after panel operations')
else console.log('scene canvas alive: OK')
await page.screenshot({ path: `${OUT}/panel-final.png` })

console.log(problems.length ? `PROBLEMS:\n${problems.join('\n')}` : 'ALL PANEL CHECKS PASSED')
await browser.close()
process.exit(problems.length ? 1 : 0)
