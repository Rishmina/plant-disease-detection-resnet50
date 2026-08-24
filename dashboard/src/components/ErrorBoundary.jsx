import { Component } from 'react'

export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('Canopy Scan crashed:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto flex min-h-screen max-w-lg flex-col items-center justify-center gap-4 px-6 text-center">
          <p className="font-mono text-[11px] uppercase tracking-widest2 text-rust">System fault</p>
          <h1 className="font-heading text-2xl text-bone">The scanner hit an unexpected error</h1>
          <p className="text-sm text-sage">
            Reloading usually clears it. If it keeps happening, check the browser console for details.
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="border border-leaf/40 bg-leaf/10 px-4 py-2 font-mono text-xs uppercase tracking-widest2 text-leaf transition-colors hover:bg-leaf/20 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-leaf"
          >
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
