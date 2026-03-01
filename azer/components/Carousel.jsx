'use client'

import { useState } from 'react'
import dynamic from 'next/dynamic'

// Lazy load slides to avoid SSR canvas issues
const SlideEEG     = dynamic(() => import('./SlideEEG'),     { ssr: false })
const SlideICA     = dynamic(() => import('./SlideICA'),     { ssr: false })
const SlidePSD     = dynamic(() => import('./SlidePSD'),     { ssr: false })
const SlideSensors = dynamic(() => import('./SlideSensors'), { ssr: false })

const SLIDES = [
  { label: 'EEG Signal',      Component: SlideEEG     },
  { label: 'ICA Components',  Component: SlideICA     },
  { label: 'PSD',             Component: SlidePSD     },
  { label: 'Sensor Map',      Component: SlideSensors },
]

export default function Carousel() {
  const [current, setCurrent] = useState(0)

  function move(dir) {
    setCurrent(v => Math.max(0, Math.min(SLIDES.length - 1, v + dir)))
  }

  return (
    <div style={{
      flex: 1, minHeight: 0,
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Nav bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px',
        height: 40,
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
        gap: 0,
      }}>
        <ArrowButton dir="left"  disabled={current === 0}                  onClick={() => move(-1)} />

        {/* Tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 2, margin: '0 12px', flex: 1 }}>
          {SLIDES.map(({ label }, i) => (
            <button
              key={label}
              onClick={() => setCurrent(i)}
              style={{
                padding: '4px 12px',
                fontSize: 11, fontWeight: 500,
                color: current === i ? 'var(--accent)' : 'var(--text-tertiary)',
                background: current === i ? 'var(--accent-soft)' : 'none',
                border: `1px solid ${current === i ? 'var(--accent-mid)' : 'transparent'}`,
                borderRadius: 'var(--r-sm)',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-disabled)', marginRight: 8 }}>
          {current + 1} / {SLIDES.length}
        </span>

        <ArrowButton dir="right" disabled={current === SLIDES.length - 1} onClick={() => move(1)} />
      </div>

      {/* Slide content */}
      <div style={{ flex: 1, minHeight: 0, position: 'relative', overflow: 'hidden' }}>
        {SLIDES.map(({ label, Component }, i) => (
          <div
            key={label}
            style={{
              position: 'absolute', inset: 0,
              padding: '16px 24px 10px',
              display: 'flex', flexDirection: 'column',
              opacity: current === i ? 1 : 0,
              pointerEvents: current === i ? 'auto' : 'none',
              transition: 'opacity 0.2s ease',
              overflow: 'hidden',
            }}
          >
            <Component />
          </div>
        ))}
      </div>
    </div>
  )
}

function ArrowButton({ dir, disabled, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        width: 28, height: 28,
        border: '1px solid var(--border)',
        borderRadius: 'var(--r-sm)',
        background: 'none',
        color: 'var(--text-tertiary)',
        cursor: disabled ? 'default' : 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        opacity: disabled ? 0.3 : 1,
        flexShrink: 0,
        transition: 'all 0.15s',
      }}
    >
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
        {dir === 'left'
          ? <path d="M7.5 2L3.5 6l4 4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
          : <path d="M4.5 2l4 4-4 4"  stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
        }
      </svg>
    </button>
  )
}