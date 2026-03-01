'use client'

import { useEffect, useRef, useState } from 'react'

const CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']

function generateSignal(length, baseFreq = 10, noiseLevel = 0.3, hasArtifact = false) {
  return Array.from({ length }, (_, i) => {
    const t = i / length
    const alpha = Math.sin(2 * Math.PI * baseFreq * t) * 0.6
    const beta  = Math.sin(2 * Math.PI * 20 * t) * 0.2
    const noise = (Math.random() - 0.5) * noiseLevel
    const artifact = hasArtifact && t > 0.3 && t < 0.4 ? Math.sin(2 * Math.PI * 2 * t) * 2.5 : 0
    return alpha + beta + noise + artifact
  })
}

function SignalRow({ label, data, artifactRange, width, height = 48 }) {
  const path = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height / 2 - v * (height / 3.5)
    return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')

  const isArtifact = artifactRange

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      borderBottom: '1px solid var(--border-0)',
      padding: '2px 0',
    }}>
      {/* Channel label */}
      <div style={{
        width: 36,
        flexShrink: 0,
        fontSize: 9,
        letterSpacing: 1,
        color: isArtifact ? 'var(--error)' : 'var(--text-3)',
        textAlign: 'right',
        paddingRight: 12,
        fontWeight: 600,
      }}>
        {label}
      </div>

      {/* Signal */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {/* Artifact highlight zone */}
          {isArtifact && (
            <rect
              x={width * 0.3} y={0}
              width={width * 0.1} height={height}
              fill="rgba(255,77,77,0.07)"
            />
          )}
          {/* Zero line */}
          <line
            x1={0} y1={height/2}
            x2={width} y2={height/2}
            stroke="var(--border-0)" strokeWidth={0.5}
          />
          {/* Signal path */}
          <path
            d={path}
            fill="none"
            stroke={isArtifact ? 'rgba(255,77,77,0.7)' : 'rgba(255,255,255,0.65)'}
            strokeWidth={0.8}
            strokeLinejoin="round"
          />
        </svg>
      </div>
    </div>
  )
}

export default function EEGViewer({ isProcessing = true }) {
  const containerRef = useRef(null)
  const [width, setWidth] = useState(900)
  const [signals, setSignals] = useState([])
  const [time, setTime] = useState(0)

  useEffect(() => {
    const obs = new ResizeObserver(entries => {
      for (const e of entries) setWidth(e.contentRect.width - 48)
    })
    if (containerRef.current) obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  // useEffect(() => {
  //   // Generate signals
  //   setSignals(CHANNELS.map((ch, i) => ({
  //     label: ch,
  //     data: generateSignal(400, 8 + i * 1.5, 0.25 + i * 0.05, i === 0 || i === 1),
  //     artifactRange: i === 0 || i === 1,
  //   })))
  // }, [])
  if (loading) {
    return <div>Loading signals...</div>
  }

  return (
    <div ref={containerRef} style={{ /* ...existing styles... */ }}>
      {/* ...existing header... */}

      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {signals.map((sig) => (
          <SignalRow
            key={sig.label}
            label={sig.label}
            data={sig.data}
            artifactRange={sig.artifactRange}
            width={width}
            height={46}
          />
        ))}
        
        {/* ...existing time cursor... */}
      </div>

      {/* ...existing time axis... */}
    </div>
  )
}
  // Animate time cursor
  useEffect(() => {
    if (!isProcessing) return
    const t = setInterval(() => setTime(v => (v + 1) % 100), 50)
    return () => clearInterval(t)
  }, [isProcessing])

  return (
    <div ref={containerRef} style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-0)',
      overflow: 'hidden',
      padding: '16px 24px',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{ fontSize: 9, letterSpacing: 3, color: 'var(--text-3)', fontWeight: 600 }}>
            EEG SIGNAL
          </span>
          <span style={{ fontSize: 9, color: 'var(--text-4)' }}>
            8 CH · 256 Hz · 60s epoch
          </span>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <LegendItem color="rgba(255,255,255,0.65)" label="RAW" />
          <LegendItem color="rgba(255,77,77,0.7)" label="ARTIFACT DETECTED" />
        </div>
      </div>

      {/* Signals */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {signals.map((sig) => (
          <SignalRow
            key={sig.label}
            label={sig.label}
            data={sig.data}
            artifactRange={sig.artifactRange}
            width={width}
            height={46}
          />
        ))}

        {/* Time cursor */}
        {isProcessing && (
          <div style={{
            position: 'absolute',
            top: 0, bottom: 0,
            left: `${time}%`,
            width: 1,
            background: 'rgba(255,255,255,0.15)',
            pointerEvents: 'none',
            transition: 'left 0.05s linear',
          }} />
        )}
      </div>

      {/* Time axis */}
      <div style={{
        display: 'flex',
        paddingLeft: 48,
        paddingTop: 6,
        borderTop: '1px solid var(--border-0)',
        marginTop: 4,
      }}>
        {[0, 10, 20, 30, 40, 50, 60].map(t => (
          <div key={t} style={{
            flex: 1,
            fontSize: 8,
            color: 'var(--text-4)',
            letterSpacing: 0.5,
          }}>
            {t}s
          </div>
        ))}
      </div>
    </div>
  )


function LegendItem({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ width: 16, height: 1, background: color }} />
      <span style={{ fontSize: 8, letterSpacing: 1.5, color: 'var(--text-3)' }}>{label}</span>
    </div>
  )
}