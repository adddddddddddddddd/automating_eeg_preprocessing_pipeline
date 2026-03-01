'use client'

import { useEffect, useRef, useState } from 'react'
import { AGENT_THOUGHTS, JOURNAL } from '@/lib/data'

export default function BottomPanel() {
  return (
    <div style={{
      height: 222, flexShrink: 0,
      display: 'flex',
      background: 'var(--surface)',
    }}>
      <AgentLog />
      <JournalLog />
    </div>
  )
}

function AgentLog() {
  const [lines, setLines]   = useState([])
  const bodyRef             = useRef(null)

  useEffect(() => {
    const timers = AGENT_THOUGHTS.map(([delay, text]) =>
      setTimeout(() => {
        setLines(v => [...v, text])
      }, delay)
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [lines])

  return (
    <div style={{ flex: 1, borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PanelHead title="Agent reasoning" live />
      <div ref={bodyRef} style={{ flex: 1, overflowY: 'auto', padding: '10px 18px', display: 'flex', flexDirection: 'column', gap: 1 }}>
        {lines.map((line, i) => (
          <div key={i} style={{
            fontFamily: 'var(--font-mono)', fontSize: 10, lineHeight: 1.7,
            color: i === lines.length - 1 ? 'var(--text-secondary)' : 'var(--text-tertiary)',
            animation: 'fadein 0.25s ease',
          }}>
            <span style={{ color: 'var(--accent)', marginRight: 6, fontSize: 9 }}>›</span>
            {line}
            {i === lines.length - 1 && <BlinkCursor />}
          </div>
        ))}
      </div>
    </div>
  )
}

function JournalLog() {
  return (
    <div style={{ width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <PanelHead title="Current run" />
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 14px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        {JOURNAL.map(run => (
          <div key={run.id} style={{
            padding: '8px 10px',
            border: `1px solid ${run.status === 'running' ? 'var(--accent-mid)' : run.status === 'error' ? 'rgba(138,117,96,0.2)' : 'var(--border)'}`,
            borderRadius: 'var(--r)',
            background: run.status === 'running' ? 'var(--accent-soft)' : run.status === 'error' ? 'var(--warn-soft)' : 'var(--surface-2)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 3 }}>
              <StatusDot status={run.status} />
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: 9, fontWeight: 500,
                color: run.status === 'running' ? 'var(--accent)' : run.status === 'error' ? 'var(--warn)' : 'var(--text-tertiary)',
              }}>
                {run.id}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-disabled)', marginLeft: 'auto' }}>
                {run.time}
              </span>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>{run.file}</div>
            {run.note && (
              <div style={{
                fontSize: 10, lineHeight: 1.5,
                color: run.status === 'error' ? 'var(--warn)' : 'var(--text-secondary)',
                borderTop: '1px solid var(--border)', paddingTop: 4, marginTop: 4,
              }}>
                {run.note}
              </div>
            )}
            {run.status === 'running' && (
              <div style={{ marginTop: 6, height: 2, background: 'var(--border)', borderRadius: 99, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: '30%', background: 'var(--accent)', borderRadius: 99, animation: 'scan 2s linear infinite', opacity: 0.5 }} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function PanelHead({ title, live }) {
  const [tick, setTick] = useState(true)
  useEffect(() => {
    if (!live) return
    const t = setInterval(() => setTick(v => !v), 2000)
    return () => clearInterval(t)
  }, [live])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 18px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
      {live && (
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', animation: 'blink 2s ease infinite' }} />
      )}
      <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-secondary)' }}>{title}</span>
    </div>
  )
}

function BlinkCursor() {
  return (
    <span style={{
      display: 'inline-block', width: 5, height: 10,
      background: 'var(--accent)', opacity: 0.7, marginLeft: 2,
      verticalAlign: 'middle', animation: 'blink 0.9s step-end infinite',
    }} />
  )
}

function StatusDot({ status }) {
  const colors = { running: 'var(--accent)', done: 'var(--text-disabled)', error: 'var(--warn)' }
  return (
    <div style={{
      width: 5, height: 5, borderRadius: '50%',
      background: colors[status], flexShrink: 0,
      animation: status === 'running' ? 'blink 1.8s ease infinite' : 'none',
    }} />
  )
}