'use client'

import { useEffect, useState } from 'react'
import { PIPELINE_STEPS } from '@/lib/data'

export default function PipelineBar() {
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    const delays = [1400, 3800, 7200, 12500, 20000]
    const timers = delays.map((d, i) =>
      setTimeout(() => setActiveStep(i + 1), d)
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <div style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '14px 24px',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        {PIPELINE_STEPS.map((step, i) => {
          const isDone    = i < activeStep
          const isRunning = i === activeStep
          return (
            <div key={step.name} style={{ display: 'flex', alignItems: 'center', flex: 1, minWidth: 0 }}>
              {/* Card */}
              <div style={{
                flex: 1, minWidth: 0,
                padding: '8px 12px',
                borderRadius: 'var(--r)',
                border: `1px solid ${isRunning ? 'var(--border)' : 'transparent'}`,
                background: isRunning ? 'var(--surface-2)' : 'transparent',
                boxShadow: isRunning ? '0 1px 4px rgba(0,0,0,0.06)' : 'none',
                transition: 'all 0.35s ease',
              }}>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 9, marginBottom: 2,
                  color: isRunning ? 'var(--accent)' : 'var(--text-disabled)',
                }}>
                  {String(i + 1).padStart(2, '0')}
                </div>
                <div style={{
                  fontSize: 11, fontWeight: 500,
                  color: isRunning ? 'var(--text-primary)' : isDone ? 'var(--text-disabled)' : 'var(--text-tertiary)',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {isDone ? '✓ ' : ''}{step.name}
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 9, marginTop: 2,
                  color: isRunning ? 'var(--text-tertiary)' : 'var(--text-disabled)',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {isRunning ? step.detail : isDone ? '' : step.detail}
                </div>
                {isRunning && (
                  <div style={{ marginTop: 6, height: 2, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: '35%', background: 'var(--accent)', borderRadius: 99, animation: 'scan 2s linear infinite', opacity: 0.6 }} />
                  </div>
                )}
              </div>

              {/* Connector */}
              {i < PIPELINE_STEPS.length - 1 && (
                <div style={{
                  width: 16, height: 1, flexShrink: 0,
                  background: isDone ? 'var(--border-mid)' : 'var(--border)',
                  transition: 'background 0.4s',
                }} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}