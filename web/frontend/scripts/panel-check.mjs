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
if ((await groupCount()) !== 3) problems.push(`expected 3 initial groups, got ${await groupCount()}`)

// 1. drag the input window by its header
const input0 = await rectOf(windowWithTab('input'))
await drag(input0.x + 120, input0.y + 14, input0.x + 420, input0.y + 90)
const input1 = await rectOf(windowWithTab('input'))
if (Math.abs(input1.x - input0.x - 300) > 20 || Math.abs(input1.y - input0.y - 76) > 20)
  problems.push(`input window did not move as expected: ${JSON.stringify({ input0, input1 })}`)
else console.log('drag window: OK')

// 2. minimize / restore the scene window
const scene = page.locator(windowWithTab('scene'))
await scene.locator('.win-btn').click()
await page.waitForTimeout(200)
const minimizedH = (await rectOf(windowWithTab('scene'))).height
if (minimizedH > 40) problems.push(`scene not minimized (height ${minimizedH})`)
else console.log('minimize: OK')
await page.screenshot({ path: `${OUT}/panel-minimized.png` })
await scene.locator('.win-btn').click()
await page.waitForTimeout(200)
if ((await rectOf(windowWithTab('scene'))).height < 100) problems.push('scene did not restore')
else console.log('restore: OK')

const dumpGroups = async (label) => {
  const tabs = await page.locator('.panel-window').evaluateAll((els) =>
    els.map((el) => el.getAttribute('data-tabs')),
  )
  console.log(`${label}:`, JSON.stringify(tabs))
}

// 3. drag the diagram tab out of the descriptor group into empty space
// (bottom-left corner is free: the input window was moved right in step 1)
const diagramTab = page.locator('.tab', { hasText: 'Diagram' })
const tabBox = await diagramTab.boundingBox()
await drag(tabBox.x + tabBox.width / 2, tabBox.y + tabBox.height / 2, 80, 860)
await dumpGroups('after detach drag')
if ((await groupCount()) !== 4) problems.push(`detach failed: ${await groupCount()} groups`)
else console.log('detach tab -> floating window: OK')
await page.screenshot({ path: `${OUT}/panel-detached.png` })

// 4. drag the floating diagram window onto the barcode group header -> merge
const diag = await rectOf('.panel-window[data-tabs="diagram"]')
const barcode = await rectOf(windowWithTab('barcode'))
await drag(diag.x + 150, diag.y + 14, barcode.x + barcode.width - 80, barcode.y + 14)
await dumpGroups('after merge drag')
if ((await groupCount()) !== 3) problems.push(`merge failed: ${await groupCount()} groups`)
else console.log('merge by dragging onto header: OK')
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
