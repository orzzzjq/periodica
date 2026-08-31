import { create } from 'zustand'

export type PanelId = 'input' | 'scene' | 'barcode' | 'diagram' | 'image'

export const PANEL_IDS: PanelId[] = ['input', 'scene', 'barcode', 'diagram', 'image']

export const PANEL_TITLES: Record<PanelId, string> = {
  input: 'Input',
  scene: 'Visualization',
  barcode: 'Barcode',
  diagram: 'Diagram',
  image: 'Image',
}

export const HEADER_H = 32
export const APP_BAR_H = 36
const MIN_W = 220
const MIN_H = 140
const STORAGE_KEY = 'periodica-panels-v1'

export interface PanelGroup {
  id: string
  x: number
  y: number
  w: number
  h: number
  z: number
  minimized: boolean
  tabs: PanelId[]
  active: PanelId
}

interface PanelState {
  groups: Record<string, PanelGroup>
  zTop: number
  dropTarget: string | null
  bringToFront: (gid: string) => void
  moveGroup: (gid: string, x: number, y: number) => void
  resizeGroup: (gid: string, w: number, h: number) => void
  toggleMinimize: (gid: string) => void
  restore: (gid: string) => void
  setActive: (gid: string, panel: PanelId) => void
  setDropTarget: (gid: string | null) => void
  mergeGroups: (src: string, dst: string) => void
  moveTab: (srcGid: string, panel: PanelId, dstGid: string) => void
  detachTab: (gid: string, panel: PanelId, x: number, y: number) => void
  resetLayout: () => void
}

let idCounter = 0
const newId = () => `g${Date.now().toString(36)}-${idCounter++}`

function clampPos(x: number, y: number, w: number): [number, number] {
  const vw = window.innerWidth
  const vh = window.innerHeight - APP_BAR_H
  return [Math.min(Math.max(x, -w + 80), vw - 60), Math.min(Math.max(y, 0), vh - HEADER_H)]
}

function defaultLayout(): Record<string, PanelGroup> {
  const vw = Math.max(window.innerWidth, 900)
  const vh = Math.max(window.innerHeight - APP_BAR_H, 500)
  const descW = 440
  const inputW = 300
  const gap = 10
  const h = vh - 2 * gap
  const groups: PanelGroup[] = [
    { id: newId(), x: gap, y: gap, w: inputW, h: Math.min(640, h), z: 1, minimized: false, tabs: ['input'], active: 'input' },
    {
      id: newId(),
      x: inputW + 2 * gap,
      y: gap,
      w: Math.max(vw - inputW - descW - 4 * gap, 400),
      h,
      z: 2,
      minimized: false,
      tabs: ['scene'],
      active: 'scene',
    },
    {
      id: newId(),
      x: vw - descW - gap,
      y: gap,
      w: descW,
      h,
      z: 3,
      minimized: false,
      tabs: ['barcode', 'diagram', 'image'],
      active: 'barcode',
    },
  ]
  return Object.fromEntries(groups.map((g) => [g.id, g]))
}

function loadLayout(): Record<string, PanelGroup> | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed?.v !== 1 || typeof parsed.groups !== 'object') return null
    const groups = parsed.groups as Record<string, PanelGroup>
    // every panel must appear exactly once
    const seen = Object.values(groups).flatMap((g) => g.tabs)
    if (seen.length !== PANEL_IDS.length || new Set(seen).size !== PANEL_IDS.length) return null
    if (!PANEL_IDS.every((p) => seen.includes(p))) return null
    for (const g of Object.values(groups)) {
      if (!g.tabs.includes(g.active)) g.active = g.tabs[0]
      g.w = Math.max(g.w, MIN_W)
      g.h = Math.max(g.h, MIN_H)
      ;[g.x, g.y] = clampPos(g.x, g.y, g.w)
    }
    return groups
  } catch {
    return null
  }
}

function saveLayout(groups: Record<string, PanelGroup>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ v: 1, groups }))
  } catch {
    /* ignore */
  }
}

export const usePanelStore = create<PanelState>((set, get) => {
  const update = (mutate: (groups: Record<string, PanelGroup>) => void) => {
    const groups = structuredClone(get().groups)
    mutate(groups)
    set({ groups })
    saveLayout(groups)
  }

  return {
    groups: loadLayout() ?? defaultLayout(),
    zTop: 10,
    dropTarget: null,

    bringToFront: (gid) => {
      const z = get().zTop + 1
      set({ zTop: z })
      update((gs) => {
        if (gs[gid]) gs[gid].z = z
      })
    },

    moveGroup: (gid, x, y) =>
      update((gs) => {
        const g = gs[gid]
        if (g) [g.x, g.y] = clampPos(x, y, g.w)
      }),

    resizeGroup: (gid, w, h) =>
      update((gs) => {
        const g = gs[gid]
        if (g) {
          g.w = Math.max(w, MIN_W)
          g.h = Math.max(h, MIN_H)
        }
      }),

    toggleMinimize: (gid) =>
      update((gs) => {
        if (gs[gid]) gs[gid].minimized = !gs[gid].minimized
      }),

    restore: (gid) => {
      const z = get().zTop + 1
      set({ zTop: z })
      update((gs) => {
        const g = gs[gid]
        if (g) {
          g.minimized = false
          g.z = z
        }
      })
    },

    setActive: (gid, panel) =>
      update((gs) => {
        const g = gs[gid]
        if (g && g.tabs.includes(panel)) g.active = panel
      }),

    setDropTarget: (gid) => set({ dropTarget: gid }),

    mergeGroups: (src, dst) => {
      const z = get().zTop + 1
      set({ zTop: z })
      update((gs) => {
        const s = gs[src]
        const d = gs[dst]
        if (!s || !d || src === dst) return
        d.tabs.push(...s.tabs)
        d.active = s.active
        d.minimized = false
        d.z = z
        delete gs[src]
      })
    },

    moveTab: (srcGid, panel, dstGid) => {
      const z = get().zTop + 1
      set({ zTop: z })
      update((gs) => {
        const s = gs[srcGid]
        const d = gs[dstGid]
        if (!s || !d || srcGid === dstGid || !s.tabs.includes(panel)) return
        s.tabs = s.tabs.filter((t) => t !== panel)
        if (s.tabs.length === 0) delete gs[srcGid]
        else if (s.active === panel) s.active = s.tabs[0]
        d.tabs.push(panel)
        d.active = panel
        d.minimized = false
        d.z = z
      })
    },

    detachTab: (gid, panel, x, y) => {
      const z = get().zTop + 1
      set({ zTop: z })
      update((gs) => {
        const g = gs[gid]
        if (!g || !g.tabs.includes(panel)) return
        if (g.tabs.length === 1) {
          ;[g.x, g.y] = clampPos(x, y, g.w)
          g.z = z
          return
        }
        g.tabs = g.tabs.filter((t) => t !== panel)
        if (g.active === panel) g.active = g.tabs[0]
        const w = Math.min(g.w, 460)
        const h = Math.min(g.h, 480)
        const [nx, ny] = clampPos(x, y, w)
        const id = newId()
        gs[id] = { id, x: nx, y: ny, w, h, z, minimized: false, tabs: [panel], active: panel }
      })
    },

    resetLayout: () => {
      const groups = defaultLayout()
      set({ groups, zTop: 10, dropTarget: null })
      saveLayout(groups)
    },
  }
})
