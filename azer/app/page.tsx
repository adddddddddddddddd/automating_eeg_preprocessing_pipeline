'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'

export default function LandingPage() {
  const router = useRouter()
  const [phase, setPhase] = useState('hidden') // hidden → in → idle → out

  useEffect(() => {
    const t1 = setTimeout(() => setPhase('in'),   100)
    const t2 = setTimeout(() => setPhase('idle'), 900)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [])

  function enter() {
    setPhase('out')
    setTimeout(() => router.push('/dashboard'), 600)
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400&family=DM+Sans:wght@400;600&display=swap');
        @keyframes ring-breathe {
          0%, 100% { transform: scale(1);    opacity: 1; }
          50%       { transform: scale(1.22); opacity: 0.35; }
        }
        @keyframes dot-pulse {
          0%, 100% { transform: scale(1);    opacity: 1; }
          50%       { transform: scale(0.82); opacity: 0.65; }
        }
      `}</style>

      <div style={{
        height: '100vh', width: '100vw',
        background: '#FAFAFA',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        fontFamily: "'DM Sans', sans-serif",
      }}>

        {/* Dot + wordmark */}
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 28,
          opacity:   phase === 'hidden' ? 0 : phase === 'out' ? 0 : 1,
          transform: phase === 'hidden' ? 'translateY(10px)' : phase === 'out' ? 'translateY(-6px)' : 'translateY(0)',
          transition: 'opacity 0.6s ease, transform 0.6s ease',
        }}>

          {/* Animated dot */}
          <div style={{ position: 'relative', width: 52, height: 52, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{
              position: 'absolute', width: 52, height: 52, borderRadius: '50%',
              border: '1px solid rgba(74,111,165,0.14)',
              animation: phase === 'idle' ? 'ring-breathe 3.2s ease-in-out infinite' : 'none',
            }} />
            <div style={{
              position: 'absolute', width: 34, height: 34, borderRadius: '50%',
              border: '1px solid rgba(74,111,165,0.2)',
              animation: phase === 'idle' ? 'ring-breathe 3.2s ease-in-out infinite 0.5s' : 'none',
            }} />
            <div style={{
              width: 10, height: 10, borderRadius: '50%', background: '#4A6FA5',
              animation: phase === 'idle' ? 'dot-pulse 3.2s ease-in-out infinite' : 'none',
            }} />
          </div>

          {/* Wordmark */}
          <div style={{ fontWeight: 600, fontSize: 17, letterSpacing: '-0.3px', color: '#111827' }}>
            Neural<span style={{ color: '#4A6FA5' }}>Prep</span>
          </div>
        </div>

        {/* Enter */}
        <button onClick={enter} style={{
          position: 'absolute', bottom: 48,
          fontFamily: "'IBM Plex Mono', monospace",
          fontSize: 10, letterSpacing: '2.5px', textTransform: 'uppercase',
          color: '#9CA3AF', background: 'none', border: 'none', cursor: 'pointer',
          opacity: phase === 'idle' ? 1 : 0,
          transform: phase === 'idle' ? 'translateY(0)' : 'translateY(5px)',
          transition: 'opacity 0.5s ease 0.5s, transform 0.5s ease 0.5s',
          padding: '8px 16px',
        }}>
          enter
        </button>
      </div>
    </>
  )
}