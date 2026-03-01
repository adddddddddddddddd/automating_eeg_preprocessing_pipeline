'use client'

import { useEffect, useRef, useState } from 'react'
import { EEG_CHANNELS, ARTIFACT_CHANNELS, genSignal } from '@/lib/data'

const W = 900, H = 44

function signalPath(data) {
  return data.map((v, i) => {
    const x = (i / (data.length - 1)) * W
    const y = H / 2 - v * (H / 3.8)
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
}

export default function SlideEEG() {
  const [signals] = useState(() =>
    EEG_CHANNELS.map((ch, i) => ({
      ch,
      isArt: ARTIFACT_CHANNELS.has(i),
      path: signalPath(genSignal(400, 8 + i * 1.5, 0.22 + i * 0.04, ARTIFACT_CHANNELS.has(i))),
    }))
  )
  const [pct, setPct] = useState(0)

  useEffect(() => {
    const t = setInterval(() => setPct(v => (v + 0.4) % 100), 50)
    return () => clearInterval(t)
  }, [])

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-secondary)' }}>EEG Signal</span>
          {['8 ch', '256 Hz', '60 s'].map(c => <MetaChip key={c} label={c} />)}
        </div>
        <div style={{ display: 'flex', gap: 14 }}>
          <LegendItem color="var(--text-secondary)" label="Raw" />
          <LegendItem color="var(--warn)" label="Artifact" />
        </div>
      </div>

      {/* Signals */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {signals.map(({ ch, isArt, path }) => (
          <div key={ch} style={{ display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--bg)' }}>
            <div style={{
              width: 34, flexShrink: 0,
              fontFamily: 'var(--font-mono)', fontSize: 9,
              color: isArt ? 'var(--warn)' : 'var(--text-tertiary)',
              textAlign: 'right', paddingRight: 10,
            }}>
              {ch}
            </div>
            <div style={{ flex: 1, overflow: 'hidden' }}>
              <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
                {isArt && <rect x={W * 0.3} y={0} width={W * 0.1} height={H} fill="var(--warn-soft)" />}
                <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="var(--border)" strokeWidth={0.5} />
                <path d={path} fill="none"
                  stroke={isArt ? 'var(--warn)' : 'var(--text-secondary)'}
                  strokeWidth={0.9} strokeLinejoin="round"
                  opacity={isArt ? 0.55 : 0.45}
                />
              </svg>
            </div>
          </div>
        ))}
        {/* Cursor */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0, left: `${pct}%`,
          width: 1, background: 'var(--accent)', opacity: 0.2,
          pointerEvents: 'none', transition: 'left 0.05s linear',
        }} />
      </div>

      {/* Time axis */}
      <div style={{ display: 'flex', paddingLeft: 34, paddingTop: 5, borderTop: '1px solid var(--border)', marginTop: 4, flexShrink: 0 }}>
        {[0, 10, 20, 30, 40, 50, 60].map(t => (
          <div key={t} style={{ flex: 1, fontFamily: 'var(--font-mono)', fontSize: 8, color: 'var(--text-disabled)' }}>
            {t} s
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