'use client'

import { createContext, useContext, useState, useEffect } from 'react'
import { useAgentSocket } from './useAgentSocket'

const SignalContext = createContext(null)

export function SignalProvider({ children }) {
  const { logs, pipelineStep, connected, done } = useAgentSocket()
  const [signals, setSignals] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (pipelineStep === 0) return
    
    setLoading(true)
    
    fetch(`http://localhost:8000/signals/1/step/${pipelineStep}/timeseries`)
      .then(res => res.json())
      .then(res => {
        const { channels, data, times } = res

        const signals = channels.map((ch, i) => ({
          label: ch,
          data: data[i],
          times: times,
          artifactRange: false, // Ajuster selon vos données
        }))

        setSignals(signals)
      })
      .catch(err => console.error('Error fetching signals:', err))
      .finally(() => setLoading(false))
  }, [pipelineStep])

  return (
    <SignalContext.Provider value={{ signals, loading, pipelineStep, logs, connected, done }}>
      {children}
    </SignalContext.Provider>
  )
}

export function useSignals() {
  const context = useContext(SignalContext)
  if (!context) {
    throw new Error('useSignals must be used within SignalProvider')
  }
  return context
}