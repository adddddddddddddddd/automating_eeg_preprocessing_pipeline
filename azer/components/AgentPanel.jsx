'use client'

import { useEffect, useRef, useState } from 'react'
import { useAgentSocket } from '@/lib/useAgentSocket'

export function AgentPanel({ runId }) {
  const { logs, steps, connected } = useAgentSocket(runId)

  return (
    <div style={{
      width: 320,
      flexShrink: 0,
      background: 'var(--surface)',
      borderLeft: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <PanelHeader label="AGENT" live={connected} />
      
      {/* Content */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 12,
      }}>
        {/* Status de connexion */}
        {runId && (
          <div style={{
            fontSize: 10,
            color: 'var(--text-tertiary)',
            fontFamily: 'var(--font-mono)',
          }}>
            {connected ? '🟢 Connecté au pipeline' : '🔴 Déconnecté'}
          </div>
        )}

        {/* Affichage des logs */}
        {logs.length > 0 ? (
          logs.map((log, idx) => (
            <div
              key={idx}
              style={{
                padding: 12,
                borderRadius: 'var(--r)',
                background: 'var(--surface-raised)',
                border: '1px solid var(--border)',
              }}
            >
              <div style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 8,
                marginBottom: 8,
              }}>
                <Brain size={14} style={{ color: ' var(--accent)', marginTop: 2 }} />
                <div style={{ flex: 1 }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    marginBottom: 4,
                  }}>
                    <span style={{
                      fontSize: 10,
                      fontWeight: 600,
                      fontFamily: 'var(--font-mono)',
                      color: log.level === 'error' ? 'var(--error)' :
                             log.level === 'warning' ? 'var(--warning)' :
                             'var(--accent)',
                    }}>
                      [{log.level.toUpperCase()}]
                    </span>
                    <span style={{
                      fontSize: 9,
                      color: 'var(--text-tertiary)',
                      fontFamily: 'var(--font-mono)',
                    }}>
                      {new Date(log.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p style={{
                    fontSize: 11,
                    lineHeight: 1.5,
                    color: 'var(--text-secondary)',
                  }}>
                    {log.message}
                  </p>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div style={{
            textAlign: 'center',
            color: 'var(--text-tertiary)',
            fontSize: 12,
            padding: '32px 0',
          }}>
            {runId ? 'En attente des logs...' : 'Aucun process en cours'}
          </div>
        )}

        {/* Affichage des steps */}
        {steps.length > 0 && (
          <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <h4 style={{
              fontSize: 10,
              fontWeight: 600,
              color: 'var(--text-tertiary)',
              letterSpacing: 1,
            }}>
              ÉTAPES DU PIPELINE
            </h4>
            {steps.map((step, idx) => (
              <div key={idx} style={{
                padding: 8,
                borderRadius: 'var(--r-sm)',
                background: 'var(--surface-raised)',
                fontSize: 11,
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{step.step_name}</span>
                  <span style={{
                    fontWeight: 600,
                    fontFamily: 'var(--font-mono)',
                    color: step.status === 'completed' ? 'var(--success)' :
                           step.status === 'error' ? 'var(--error)' :
                           'var(--warning)',
                  }}>
                    {step.status}
                  </span>
                </div>
                {step.progress && (
                  <div style={{
                    width: '100%',
                    height: 2,
                    background: 'var(--border)',
                    borderRadius: 1,
                    marginTop: 6,
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      height: '100%',
                      background: 'var(--accent)',
                      width: `${step.progress}%`,
                      transition: 'width 0.3s',
                    }} />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function PanelHeader({ label, live }) {
  const [tick, setTick] = useState(true)
  useEffect(() => {
    if (!live) return
    const t = setInterval(() => setTick(v => !v), 900)
    return () => clearInterval(t)
  }, [live])

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '8px 16px',
      borderBottom: '1px solid var(--border)',
      flexShrink: 0,
    }}>
      {live && (
        <div style={{
          width: 5, height: 5, borderRadius: '50%',
          background: 'var(--accent)',
          opacity: tick ? 1 : 0.2,
          transition: 'opacity 0.3s',
        }} />
      )}
      <span style={{ fontSize: 9, letterSpacing: 3, color: 'var(--text-tertiary)', fontWeight: 600 }}>
        {label}
      </span>
    </div>
  )
}