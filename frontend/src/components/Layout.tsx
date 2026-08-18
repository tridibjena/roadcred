import type { ReactNode } from 'react'

export type PageKey = 'inference' | 'comparison' | 'robustness'

const TABS: { key: PageKey; label: string; hint: string }[] = [
  { key: 'inference', label: 'Inference', hint: 'Segment an image' },
  { key: 'comparison', label: 'Model Comparison', hint: 'Ablations and trade-offs' },
  { key: 'robustness', label: 'Robustness', hint: 'Generalization and corruption' },
]

export function Layout({
  page,
  onNavigate,
  children,
}: {
  page: PageKey
  onNavigate: (page: PageKey) => void
  children: ReactNode
}) {
  return (
    <div className="min-h-screen bg-slate-950">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-8 gap-y-3 px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-slate-50">RoadSense</h1>
            <p className="text-xs text-slate-400">
              7-class segmentation of unstructured Indian road scenes · IDD
            </p>
          </div>
          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                onClick={() => onNavigate(tab.key)}
                title={tab.hint}
                className={`rounded-md px-3 py-1.5 text-sm transition ${
                  page === tab.key
                    ? 'bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/40'
                    : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>
    </div>
  )
}

export function Card({
  title,
  subtitle,
  children,
  className = '',
}: {
  title?: string
  subtitle?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-slate-900/40 p-5 ${className}`}>
      {title && <h2 className="text-sm font-semibold text-slate-200">{title}</h2>}
      {subtitle && <p className="mt-1 text-xs leading-relaxed text-slate-400">{subtitle}</p>}
      <div className={title ? 'mt-4' : ''}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 px-4 py-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="mt-1 font-mono text-xl text-slate-100">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </div>
  )
}

export function Message({ kind, children }: { kind: 'error' | 'info'; children: ReactNode }) {
  const styles =
    kind === 'error'
      ? 'border-rose-500/40 bg-rose-500/10 text-rose-200'
      : 'border-sky-500/30 bg-sky-500/10 text-sky-200'
  return <div className={`rounded-lg border px-4 py-3 text-sm ${styles}`}>{children}</div>
}
