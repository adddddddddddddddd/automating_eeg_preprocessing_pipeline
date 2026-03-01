'use client'

import { useEffect, useRef } from 'react'
import { genPSD } from '@/lib/data'

const PSD_PAIRS = [['Fp1','Fp2'],['F3','F4'],['C3','C4'],['P3','P4']]
const BANDS = [
  { lbl: 'δ', f: 2  },
  { lbl: 'θ', f: 6  },
  { lbl: 'α', f: 10 },
  { lbl: 'β', f: 20 },
  { lbl: 'γ', f: 35 },
]

function PSDCanvas({ pre, post }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    canvas.width  = canvas.offsetWidth  || 300
    canvas.height = canvas.offsetHeight || 80
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height

    ctx.clearRect(0, 0, W, H)

    const fx = f => (f / 40) * W
    const py = p => H - p * (H - 10) - 5

    // Grid
    ctx.strokeStyle = 'rgba(0,0,0,0.04)'; ctx.lineWidth = 1
    ;[0.25, 0.5, 0.75].forEach(v => {
      ctx.beginPath(); ctx.moveTo(0, py(v)); ctx.lineTo(W, py(v)); ctx.stroke()
    })
    BANDS.forEach(({ f }) => {
      ctx.beginPath(); ctx.moveTo(fx(f), 0); ctx.lineTo(fx(f), H); ctx.stroke()
    })

    // Pre curve
    ctx.beginPath()
    pre.forEach((p, i) => { i === 0 ? ctx.moveTo(fx(i+1), py(p)) : ctx.lineTo(fx(i+1), py(p)) })
    ctx.strokeStyle = 'rgba(107,114,128,0.4)'; ctx.lineWidth = 1.2; ctx.stroke()

    // Post curve + fill
    ctx.beginPath()
    post.forEach((p, i) => { i === 0 ? ctx.moveTo(fx(i+1), py(p)) : ctx.lineTo(fx(i+1), py(p)) })
    ctx.strokeStyle = 'rgba(74,111,165,0.7)'; ctx.lineWidth = 1.5; ctx.stroke()
    ctx.lineTo(fx(40), H); ctx.lineTo(fx(1), H); ctx.closePath()
    ctx.fillStyle = 'rgba(74,111,165,0.06)'; ctx.fill()

    // Band labels
    ctx.fillStyle = 'rgba(156,163,175,0.9)'; ctx.font = '8px IBM Plex Mono'
    BANDS.forEach(({ lbl, f }) => ctx.fillText(lbl, fx(f) + 2, H - 3))
  }, [pre, post])

  return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }} />
}

export default function SlidePSD() {
  const pairs = PSD_PAIRS.map(([ch1, ch2]) => ({
    label: `${ch1} / ${ch2}`,
    pre:  genPSD(0.05, 0.25),
    post: genPSD(0.02, 0.10),
  }))

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-secondary)' }}>Power Spectral Density</span>
          {['Welch method', '1–40 Hz'].map(c => <MetaChip key={c} label={c} />)}
        </div>
        <div style={{ display: 'flex', gap: 14 }}>
          <LegendItem color="rgba(107,114,128,0.5)" label="Pre-filter" />
          <LegendItem color="var(--accent)" label="Post-filter" />
        </div>
      </div>

      {/* Grid */}
      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 12,
        overflow: 'hidden',
      }}>
        {pairs.map(({ label, pre, post }) => (
          <div key={label} style={{
            background: 'var(--surface-2)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--r)',
            padding: 12,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}>
            <div style={{ fontSize: 10, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 8, flexShrink: 0 }}>
              {label}
            </div>
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <PSDCanvas pre={pre} post={post} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function MetaChip({ label }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-tertiary)',
      background: 'var(--bg)', border: '1px solid var(--border)',
      padding: '2px 8px', borderRadius: 99,
    }}>
      {label}
    </span>
  )
}

function LegendItem({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, color: 'var(--text-tertiary)' }}>
      <div style={{ width: 14, height: 1.5, borderRadius: 99, background: color }} />
      {label}
    </div>
  )
}