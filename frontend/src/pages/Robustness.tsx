import { useEffect, useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Card, Message, Stat } from '../components/Layout'
import { Table } from '../components/Table'
import { figureUrl, getResults, type ResultsPayload } from '../api'

const AXIS = { stroke: '#64748b', fontSize: 11 }
const GRID = '#1e293b'
const TOOLTIP = {
  contentStyle: { background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 },
}

const SPLIT_LABELS: Record<string, string> = {
  official: "IDD's own split (drive-disjoint)",
  sequence: 'Held-out drive sequences',
  frame: 'Random frame split (leaky control)',
}

export default function Robustness() {
  const [data, setData] = useState<ResultsPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getResults().then(setData).catch((e: Error) => setError(e.message))
  }, [])

  const splitChart = useMemo(() => {
    const rows = (data?.tables.split_experiment ?? []).filter((r) => r.miou != null)
    return rows.map((r) => {
      const mode = String(r.data ?? '').replace('level1_', '') || String(r.experiment ?? '')
      return {
        mode,
        label: SPLIT_LABELS[mode] ?? mode,
        miou: r.miou as number,
        leaky: mode === 'frame',
      }
    })
  }, [data])

  const corruptionChart = useMemo(() => {
    const rows = data?.tables.corruption_robustness ?? []
    const severities = [...new Set(rows.filter((r) => r.severity !== 0).map((r) => r.severity as number))].sort()
    const names = [...new Set(rows.filter((r) => r.corruption !== 'clean').map((r) => String(r.corruption)))]
    const series = severities.map((severity) => {
      const point: Record<string, number> = { severity }
      for (const name of names) {
        const match = rows.find((r) => r.corruption === name && r.severity === severity)
        if (match?.miou != null) point[name] = match.miou as number
      }
      return point
    })
    return { series, names }
  }, [data])

  if (error) return <Message kind="error">{error}</Message>
  if (!data) return <p className="text-sm text-slate-400">Loading results…</p>

  const leak = splitChart.find((s) => s.leaky)
  const honest = splitChart.find((s) => s.mode === 'official') ?? splitChart.find((s) => s.mode === 'sequence')
  const inflation = leak && honest ? leak.miou - honest.miou : null

  const corruptionRows = data.tables.corruption_robustness ?? []
  const clean = corruptionRows.find((r) => r.corruption === 'clean')?.miou as number | undefined
  const recovered = corruptionRows.filter((r) => r.recovered != null && r.corruption !== 'clean')
  const meanRecovered = recovered.length
    ? recovered.reduce((sum, r) => sum + (r.recovered as number), 0) / recovered.length
    : null

  const hasFigure = (name: string) => data.figures.includes(name)

  return (
    <div className="space-y-6">
      {!splitChart.length && !corruptionRows.length && (
        <Message kind="info">
          No robustness tables yet. Run{' '}
          <code className="font-mono text-xs">python scripts/run_ablations.py --suite split</code>.
        </Message>
      )}

      {splitChart.length > 0 && (
        <Card
          title="Does a random split overstate generalization?"
          subtitle="The same model trained on three datasets that differ only in how frames were assigned to train/val. IDD groups frames into drive sequences; a random frame split puts 94.6% of validation frames in a drive that also appears in training, so it scores memorisation alongside generalization."
        >
          {inflation !== null && (
            <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Stat label="Drive-disjoint mIoU" value={honest!.miou.toFixed(4)} hint="the honest number" />
              <Stat label="Random-split mIoU" value={leak!.miou.toFixed(4)} hint="94.6% of val frames leaked" />
              <Stat
                label="Inflation"
                value={`${inflation >= 0 ? '+' : ''}${inflation.toFixed(4)}`}
                hint="what a naive split would have claimed"
              />
            </div>
          )}
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={splitChart} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="mode" tick={AXIS} />
              <YAxis tick={AXIS} domain={['auto', 'auto']} />
              <Tooltip {...TOOLTIP}
                formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : String(value))}
                labelFormatter={(label) => SPLIT_LABELS[String(label)] ?? String(label)} />
              <Bar dataKey="miou" radius={[4, 4, 0, 0]}>
                {splitChart.map((entry, index) => (
                  <Cell key={index} fill={entry.leaky ? '#f87171' : '#34d399'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="mt-3 text-xs text-slate-500">
            Red is the leaky control. It is reported to be argued against, never as a headline number.
          </p>
        </Card>
      )}

      {corruptionChart.series.length > 0 && (
        <Card
          title="Corruption robustness"
          subtitle="Test-time corruptions standing in for adverse driving conditions. These never appear in training augmentation, so the drop measures genuine distribution shift."
        >
          {clean != null && (
            <div className="mb-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <Stat label="Clean mIoU" value={clean.toFixed(4)} />
              <Stat
                label="Worst corruption"
                value={Math.min(...corruptionRows.filter((r) => r.corruption !== 'clean').map((r) => r.miou as number)).toFixed(4)}
              />
              {meanRecovered !== null && (
                <Stat
                  label="Recovered by BN adapt"
                  value={`${meanRecovered >= 0 ? '+' : ''}${meanRecovered.toFixed(4)}`}
                  hint="mean mIoU, no labels or gradients"
                />
              )}
            </div>
          )}
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={corruptionChart.series} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis dataKey="severity" tick={AXIS}
                label={{ value: 'severity', position: 'insideBottom', offset: -4, fill: '#64748b', fontSize: 11 }} />
              <YAxis tick={AXIS} domain={[0, 'auto']} />
              <Tooltip {...TOOLTIP} formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : String(value))} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {corruptionChart.names.map((name, index) => (
                <Line key={name} type="monotone" dataKey={name} dot={{ r: 2 }} strokeWidth={1.8}
                  stroke={['#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa',
                           '#fb923c', '#22d3ee', '#e879f9', '#4ade80', '#f87171'][index % 10]} />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4">
            <Table rows={corruptionRows} columns={['corruption', 'severity', 'miou', 'miou_bn_adapted', 'recovered', 'retention']} />
          </div>
        </Card>
      )}

      {(hasFigure('confusion_matrix.png') || hasFigure('worst_cases.png')) && (
        <Card title="Error analysis" subtitle="Which classes get confused, and what the hardest frames look like.">
          <div className="space-y-4">
            {['confusion_matrix.png', 'iou_vs_frequency.png', 'worst_cases.png']
              .filter(hasFigure)
              .map((name) => (
                <img key={name} src={figureUrl(name)} alt={name}
                  className="w-full rounded-lg border border-slate-800 bg-white" />
              ))}
          </div>
        </Card>
      )}
    </div>
  )
}
