'use client'

import { useEffect, useRef } from 'react'
import { SENSOR_DATA } from '@/lib/data'

const ELECTRODE_POSITIONS = [
  { name:'Fp1', x:-.27, y:-.77, bad:true  },
  { name:'Fp2', x: .27, y:-.77, bad:false },
  { name:'F7',  x:-.63, y:-.41, bad:false },
  { name:'F3',  x:-.35, y:-.38, bad:false },
  { name:'Fz',  x:  0,  y:-.38, bad:false },
  { name:'F4',  x: .35, y:-.38, bad:false },
  { name:'F8',  x: .63, y:-.41, bad:false },
  { name:'T7',  x:-.78, y:  0,  bad:false },
  { name:'C3',  x:-.38, y:  0,  bad:false },
  { name:'Cz',  x:  0,  y:  0,  bad:false },
  { name:'C4',  x: .38, y:  0,  bad:true  },
  { name:'T8',  x: .78, y:  0,  bad:false },
  { name:'P3',  x:-.35, y: .38, bad:false },
  { name:'Pz',  x:  0,  y: .38, bad:false },
  { name:'P4',  x: .35, y: .38, bad:true  },
  { name:'O1',  x:-.2,  y: .72, bad:false },
  { name:'Oz',  x:  0,  y: .77, bad:false },
  { name:'O2',  x: .2,  y: .72, bad:false },
]

function TopoMap() {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    canvas.width  = canvas.offsetWidth  || 300
    canvas.height = canvas.offsetHeight || 200
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const cx = W / 2, cy = H / 2
    const r = Math.min(W, H) * 0.42

    ctx.clearRect(0, 0, W, H)

    // Head circle
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2)
    ctx.fillStyle = '#F9FAFB'; ctx.fill()
    ctx.strokeStyle = '#CDD2DA'; ctx.lineWidth = 1.5; ctx.stroke()

    // Nose
    ctx.beginPath()
    ctx.moveTo(cx - 8, cy - r + 4)
    ctx.lineTo(cx, cy - r - 10)
    ctx.lineTo(cx + 8, cy - r + 4)
    ctx.strokeStyle = '#CDD2DA'; ctx.lineWidth = 1.2; ctx.stroke()

    // Ears
    ;[-1, 1].forEach(side => {
      ctx.beginPath(); ctx.arc(cx + side * (r + 2), cy, 5, 0, Math.PI * 2)
      ctx.strokeStyle = '#CDD2DA'; ctx.lineWidth = 1; ctx.stroke()
    })

    // Electrodes
    ELECTRODE_POSITIONS.forEach(({ name, x, y, bad }) => {
      const ex = cx + x * r, ey = cy + y * r
      ctx.beginPath(); ctx.arc(ex, ey, 4.5, 0, Math.PI * 2)
      ctx.fillStyle   = bad ? 'rgba(138,117,96,0.7)' : 'rgba(74,111,165,0.5)'; ctx.fill()
      ctx.strokeStyle = bad ? 'rgba(138,117,96,0.9)' : 'rgba(74,111,165,0.8)'
      ctx.lineWidth = 1; ctx.stroke()
      ctx.fillStyle = bad ? '#8A7560' : '#374B6E'
      ctx.font = '7px IBM Plex Mono'
      ctx.fillText(name, ex + 5, ey + 3)
    })
  }, [])

  return (
    <canvas
      ref={ref}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}

export default function SlideSensors() {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-secondary)' }}>Sensor Map</span>
          {['64 electrodes', '10-20 system'].map(c => <MetaChip key={c} label={c} />)}
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <MetaChip label="62 nominal" color="var(--accent)"  borderColor="var(--accent-mid)" />
          <MetaChip label="2 degraded" color="var(--warn)"    borderColor="rgba(138,117,96,0.3)" />
        </div>
      </div>

      {/* Layout */}
      <div style={{ flex: 1, display: 'flex', gap: 16, overflow: 'hidden' }}>
        {/* Topo */}
        <div style={{
          flex: 1,
          background: 'var(--surface-2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--r)',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <TopoMap />
        </div>

        {/* Impedance table */}
        <div style={{
          width: 220, flexShrink: 0,
          background: 'var(--surface-2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--r)',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}>
          {/* Table head */}
          <div style={{ display: 'flex', padding: '6px 12px', borderBottom: '1px solid var(--border)', gap: 8 }}>
            {['Ch', 'Imp. (kΩ)', ''].map((h, i) => (
              <span key={i} style={{
                fontSize: 9, fontWeight: 500, color: 'var(--text-tertiary)',
                flex: i === 0 ? undefined : i === 1 ? 1 : undefined,
                width: i === 0 ? 28 : i === 2 ? 50 : undefined,
              }}>
                {h}
              </span>
            ))}
          </div>
          {/* Rows */}
          <div style={{ flex: 1, overflowY: 'auto' }}>
            {SENSOR_DATA.map(s => (
              <div key={s.name} style={{
                display: 'flex', alignItems: 'center',
                padding: '5px 12px', gap: 8,
                borderBottom: '1px solid var(--bg)',
              }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-secondary)', width: 28, flexShrink: 0 }}>
                  {s.name}
                </span>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: 9, flex: 1,
                  color: s.ok ? 'var(--accent)' : 'var(--warn)',
                }}>
                  {s.imp.toFixed(1)}
                </span>
                <div style={{ width: 50 }}>
                  <div style={{ height: 3, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%',
                      width: `${Math.min(100, s.imp / 25 * 100)}%`,
                      background: s.ok ? 'var(--accent)' : 'var(--warn)',
                      borderRadius: 99,
                    }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function MetaChip({ label, color, borderColor }) {
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 9,
      color: color || 'var(--text-tertiary)',
      background: 'var(--bg)',
      border: `1px solid ${borderColor || 'var(--border)'}`,
      padding: '2px 8px', borderRadius: 99,
    }}>
      {label}
    </span>
  )
}