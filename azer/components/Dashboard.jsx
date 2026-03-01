'use client'

import { useState } from 'react'
import Sidebar          from './Sidebar'
import { AgentPanel }   from './AgentPanel'
import PipelineBar      from './PipelineBar'
import Carousel         from './Carousel'
import BottomPanel      from './BottomPanel'
import OnboardingModal  from './OnboardingModal'

export default function Dashboard() {
  const [toastVisible,    setToastVisible]    = useState(false)
  const [onboardingOpen,  setOnboardingOpen]  = useState(false)
  const [activeRunId,     setActiveRunId]     = useState(null)
  const [activeRun,       setActiveRun]       = useState({
    id: 'RUN-004',
    file: 'sub-01_task-rest_eeg.edf',
    status: 'processing',
  })

  function handleExport() {
    setToastVisible(true)
    setTimeout(() => setToastVisible(false), 2800)
  }

  async function handleStart({ dataset, subject, task }) {
    try {
      const response = await fetch(
        `http://localhost:8000/run?dataset_name=${dataset.name}&subject_id=${subject}&task_name=${task}`,
        { method: 'POST' }
      )
      const data = await response.json()
      
      setActiveRunId(data.run_id)
      setActiveRun({
        id: `RUN-${String(data.run_id).padStart(3, '0')}`,
        file: `${subject}_task-${task}_eeg.edf`,
        status: 'processing',
        dataset: dataset.name,
      })
    } catch (error) {
      console.error('Failed to start run:', error)
    }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* Sidebar */}
      <Sidebar
        onExport={handleExport}
        onNewRun={() => setOnboardingOpen(true)}
      />

      {/* Main */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Top nav */}
        <nav style={{
          height: 48, flexShrink: 0,
          background: 'var(--surface)',
          borderBottom: '1px solid var(--border)',
          display: 'flex', alignItems: 'center',
          padding: '0 24px', gap: 16,
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-tertiary)' }}>
            {activeRun.id}
          </span>
          <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            {activeRun.file}
          </span>
          {activeRun.dataset && (
            <>
              <div style={{ width: 1, height: 16, background: 'var(--border)' }} />
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{activeRun.dataset}</span>
            </>
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 7 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent)', animation: 'blink 2s ease infinite' }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>
              {activeRun.status}
            </span>
          </div>
        </nav>

        <PipelineBar />
        <Carousel />
        <BottomPanel />
      </div>

      {/* Agent Panel */}

      {/* Onboarding modal */}
      <OnboardingModal
        open={onboardingOpen}
        onClose={() => setOnboardingOpen(false)}
        onStart={handleStart}
      />

      {/* Export toast */}
      <div style={{
        position: 'fixed', bottom: 24, left: 260,
        background: 'var(--text-primary)', color: 'white',
        fontSize: 12, padding: '9px 16px', borderRadius: 'var(--r)',
        opacity: toastVisible ? 1 : 0,
        transition: 'opacity 0.2s',
        pointerEvents: 'none',
        zIndex: 100,
        fontFamily: 'var(--font-sans)',
      }}>
        ✓ Results exported — {activeRun.file.replace('.edf', '_features.npy')}
      </div>
    </div>
  )
}