import './globals.css'

export const metadata = {
  title: 'NeuralPrep — Agent Monitor',
  description: 'EEG preprocessing agent monitoring interface',
}
import { SignalProvider } from '@/lib/SignalContext'

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <SignalProvider>
          {children}
        </SignalProvider>
      </body>
    </html>
  )
}