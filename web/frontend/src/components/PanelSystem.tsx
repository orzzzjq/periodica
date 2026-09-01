import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import {
  APP_BAR_H,
  HEADER_H,
  PANEL_IDS,
  PANEL_TITLES,
  usePanelStore,
  type PanelGroup,
  type PanelId,
} from '../panelStore'

// Hit-test a viewport point against the other groups (store rects, not DOM,
// so the window being dragged never occludes the target). Returns the
// topmost matching group id. Group coordinates are host-relative; the host
// sits below the app bar.
function hitTest(x: number, clientY: number, exclude: string, headerOnly: boolean): string | null {
  const y = clientY - APP_BAR_H
  const groups = usePanelStore.getState().groups
  let best: PanelGroup | null = null
  for (const g of Object.values(groups)) {
    if (g.id === exclude || g.minimized) continue // minimized groups live in the app bar
    const h = headerOnly ? HEADER_H : g.h
    const inside = x >= g.x && x <= g.x + g.w && y >= g.y && y <= g.y + h
    if (inside && (!best || g.z > best.z)) best = g
  }
  return best?.id ?? null
}

function startDrag(
  e: React.PointerEvent,
  onMove: (dx: number, dy: number, ev: PointerEvent) => void,
  onUp: (ev: PointerEvent) => void,
) {
  const sx = e.clientX
  const sy = e.clientY
  const move = (ev: PointerEvent) => onMove(ev.clientX - sx, ev.clientY - sy, ev)
  const up = (ev: PointerEvent) => {
    window.removeEventListener('pointermove', move)
    window.removeEventListener('pointerup', up)
    onUp(ev)
  }
  window.addEventListener('pointermove', move)
  window.addEventListener('pointerup', up)
}

interface DragGhost {
  title: string
  x: number
  y: number
}

function GroupWindow({
  group,
  holders,
  headerExtra,
}: {
  group: PanelGroup
  holders: Record<PanelId, HTMLDivElement>
  headerExtra?: ReactNode
}) {
  const bringToFront = usePanelStore((s) => s.bringToFront)
  const moveGroup = usePanelStore((s) => s.moveGroup)
  const resizeGroup = usePanelStore((s) => s.resizeGroup)
  const toggleMinimize = usePanelStore((s) => s.toggleMinimize)
  const setActive = usePanelStore((s) => s.setActive)
  const setDropTarget = usePanelStore((s) => s.setDropTarget)
  const mergeGroups = usePanelStore((s) => s.mergeGroups)
  const moveTab = usePanelStore((s) => s.moveTab)
  const detachTab = usePanelStore((s) => s.detachTab)
  const isDropTarget = usePanelStore((s) => s.dropTarget === group.id)

  const bodyRef = useRef<HTMLDivElement>(null)
  const [ghost, setGhost] = useState<DragGhost | null>(null)

  // Reparent the stable content holders of this group's tabs into the body.
  useEffect(() => {
    const body = bodyRef.current
    if (!body) return
    for (const id of group.tabs) {
      const el = holders[id]
      if (el.parentElement !== body) body.appendChild(el)
      el.style.display = id === group.active ? '' : 'none'
    }
  })

  // Let width-tracking Plotly plots (useResizeHandler) follow panel resizes.
  useEffect(() => {
    const body = bodyRef.current
    if (!body) return
    let raf = 0
    const obs = new ResizeObserver(() => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    })
    obs.observe(body)
    return () => {
      obs.disconnect()
      cancelAnimationFrame(raf)
    }
  }, [])

  const onHeaderPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest('.tab, .win-btn, .header-extra')) return
    e.preventDefault()
    bringToFront(group.id)
    const { x, y } = group
    startDrag(
      e,
      (dx, dy, ev) => {
        moveGroup(group.id, x + dx, y + dy)
        setDropTarget(hitTest(ev.clientX, ev.clientY, group.id, true))
      },
      (ev) => {
        const target = hitTest(ev.clientX, ev.clientY, group.id, true)
        setDropTarget(null)
        if (target) mergeGroups(group.id, target)
      },
    )
  }

  const onTabPointerDown = (panel: PanelId) => (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    bringToFront(group.id)
    setActive(group.id, panel)
    let dragging = false
    startDrag(
      e,
      (dx, dy, ev) => {
        if (!dragging && Math.hypot(dx, dy) < 5) return
        dragging = true
        setGhost({ title: PANEL_TITLES[panel], x: ev.clientX, y: ev.clientY })
        const over = hitTest(ev.clientX, ev.clientY, '', false)
        setDropTarget(over !== group.id ? over : null)
      },
      (ev) => {
        setGhost(null)
        setDropTarget(null)
        if (!dragging) return
        const target = hitTest(ev.clientX, ev.clientY, '', false)
        if (target === group.id) return
        if (target) moveTab(group.id, panel, target)
        else detachTab(group.id, panel, ev.clientX - 60, ev.clientY - APP_BAR_H - HEADER_H / 2)
      },
    )
  }

  const onResizePointerDown = (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    bringToFront(group.id)
    const { w, h } = group
    startDrag(
      e,
      (dx, dy) => resizeGroup(group.id, w + dx, h + dy),
      () => {},
    )
  }

  if (group.minimized) return null // shown as an app-bar chip instead

  return (
    <div
      className={`panel-window${isDropTarget ? ' drop-target' : ''}`}
      style={{ left: group.x, top: group.y, width: group.w, height: group.h, zIndex: group.z }}
      onPointerDown={() => bringToFront(group.id)}
      data-group-id={group.id}
      data-tabs={group.tabs.join(',')}
    >
      <div className="panel-header" onPointerDown={onHeaderPointerDown}>
        <div className="panel-tabs">
          {group.tabs.map((id) => (
            <button
              key={id}
              className={`tab${id === group.active ? ' active' : ''}`}
              onPointerDown={onTabPointerDown(id)}
            >
              {PANEL_TITLES[id]}
            </button>
          ))}
          {headerExtra && <div className="header-extra">{headerExtra}</div>}
        </div>
        <button className="win-btn" title="minimize to app bar" onClick={() => toggleMinimize(group.id)}>
          —
        </button>
      </div>
      <div className="panel-body" ref={bodyRef} />
      <div className="resize-handle" onPointerDown={onResizePointerDown} />
      {ghost &&
        createPortal(
          <div className="tab-ghost" style={{ left: ghost.x + 8, top: ghost.y + 8 }}>
            {ghost.title}
          </div>,
          document.body,
        )}
    </div>
  )
}

export default function PanelHost({
  contents,
  headerExtras = {},
}: {
  contents: Record<PanelId, ReactNode>
  // rendered in a group's tab bar, after the last tab, while that panel is active
  headerExtras?: Partial<Record<PanelId, ReactNode>>
}) {
  const groups = usePanelStore((s) => s.groups)

  // One stable, never-reparented-by-React DOM node per panel: the content is
  // portaled into it exactly once, and GroupWindows move the node itself
  // between bodies. This keeps the WebGL canvas and Plotly divs alive across
  // tab moves and group merges.
  const holders = useMemo(() => {
    const map = {} as Record<PanelId, HTMLDivElement>
    for (const id of PANEL_IDS) {
      const el = document.createElement('div')
      el.className = 'panel-content-holder'
      map[id] = el
    }
    return map
  }, [])

  const restore = usePanelStore((s) => s.restore)
  const minimized = Object.values(groups)
    .filter((g) => g.minimized)
    .sort((a, b) => a.id.localeCompare(b.id))

  return (
    <>
      <div className="app-bar">
        <span className="app-brand">Periodica</span>
        <div className="app-bar-chips">
          {minimized.map((g) => (
            <button key={g.id} className="chip" title="restore" onClick={() => restore(g.id)}>
              {g.tabs.map((t) => PANEL_TITLES[t]).join(' · ')}
            </button>
          ))}
        </div>
      </div>
      <div className="panel-host">
        {Object.values(groups).map((g) => (
          <GroupWindow key={g.id} group={g} holders={holders} headerExtra={headerExtras[g.active]} />
        ))}
        {PANEL_IDS.map((id) => createPortal(contents[id], holders[id]))}
      </div>
    </>
  )
}
