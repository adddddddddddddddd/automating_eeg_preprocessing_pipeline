import { useEffect, useState, useRef } from 'react'

export function useAgentSocket(runId) {
  const [logs, setLogs] = useState([])
  const [steps, setSteps] = useState([])
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    if (!runId) return

    const websocket = new WebSocket(`ws://localhost:8000/ws/run/${runId}`)
    
    websocket.onopen = () => {
      console.log('WebSocket connected')
      setConnected(true)
    }
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      switch (data.type) {
        case 'connected':
          console.log('Pipeline started:', data.message)
          break
          
        case 'log':
          setLogs(prev => [...prev, {
            message: data.message,
            level: data.level,
            timestamp: data.timestamp
          }])
          break
          
        case 'step_update':
          setSteps(prev => {
            const existing = prev.find(s => s.step_name === data.step_name)
            if (existing) {
              return prev.map(s => 
                s.step_name === data.step_name ? { ...s, ...data } : s
              )
            }
            return [...prev, data]
          })
          break
          
        case 'graph':
          console.log('Graph data received:', data.graph_type)
          // Handle graph data
          break
          
        default:
          console.log('Unknown message type:', data.type)
      }
    }
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnected(false)
    }
    
    websocket.onclose = () => {
      console.log('WebSocket disconnected')
      setConnected(false)
    }
    
    wsRef.current = websocket
    
    return () => {
      websocket.close()
    }
  }, [runId])

  return { logs, steps, connected, ws: wsRef.current }
}