import { useEffect, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer,
  Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { Card, Message } from '../components/Layout'
import { Table } from '../components/Table'
import { getResults, type ResultsPayload, type Row } from '../api'

const AXIS = { stroke: '#64748b', fontSize: 11 }
const GRID = '#1e293b'
const TOOLTIP = {
  contentStyle: { background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 12 },
}

export default function ModelComparison() {
  const [data, setData] = useState<ResultsPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getResults().then(setData).catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <Message kind="error">{error}</Message>
  if (!data) return <p className="text-sm text-slate-400">Loading results…</p>

  const loss = (data.tables.loss_ablation ?? []).filter((r) => r.miou != null)
  const arch = (data.tables.architecture_comparison ?? []).filter((r) => r.miou != null)
  const compression = data.tables.compression ?? []
  const pretrain = (data.tables.pretrain_ablation ?? []).filter((r) => r.miou != null)

  const bestLoss = loss.length
    ? loss.reduce((a, b) => ((a.miou as number) > (b.miou as number) ? a : b))
    : null

  const lossChart = loss.map((r) => ({
    name: String(r.loss ?? r.experiment),
    miou: r.miou as number,
    best: bestLoss ? r.experiment === bestLoss.experiment : false,
  }))

  const archChart = arch.map((r) => ({
    name: String(r.experiment ?? r.architecture),
    params: ((r.params as number) ?? 0) / 1e6,
    miou: r.miou as number,
    seconds: (r.train_seconds as number) ?? 0,
  }))

  return (
    <div className="space-y-6">
      {!loss.length && !arch.length && (
        <Message kind="info">
          No ablation tables yet. Run{' '}
          <code className="font-mono text-xs">python scripts/run_ablations.py --suite loss</code>.
        </Message>
      )}

      {loss.length > 0 && (
        <Card
          title="Loss ablation"
          subtitle="Identical architecture, schedule, augmentation, split and seed — only the loss differs. IDD Lite is imbalanced (living-thing is 1.3% of pixels), so the region and boundary losses have something real to fix."
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={lossChart} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={AXIS} />
              <YAxis tick={AXIS} domain={['auto', 'auto']} label={{ value: 'mIoU', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
              <Tooltip {...TOOLTIP} formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : String(value))} />
              <Bar dataKey="miou" radius={[4, 4, 0, 0]}>
                {lossChart.map((entry, index) => (
                  <Cell key={index} fill={entry.best ? '#34d399' : '#38bdf8'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4">
            <Table
              rows={loss}
              columns={['experiment', 'loss', 'seed', 'miou', 'mean_acc', 'pixel_acc', 'epochs_ran', 'train_seconds']}
              highlight={bestLoss ? loss.indexOf(bestLoss) : undefined}
            />
          </div>
        </Card>
      )}

      {arch.length > 0 && (
        <Card
          title="Architecture comparison"
          subtitle="Accuracy against model size at a fixed encoder, loss and schedule. Up and to the left is better."
        >
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 8, right: 24, bottom: 20, left: 0 }}>
              <CartesianGrid stroke={GRID} />
              <XAxis type="number" dataKey="params" name="params" unit="M" tick={AXIS}
                label={{ value: 'parameters (M)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
              <YAxis type="number" dataKey="miou" name="mIoU" tick={AXIS} domain={['auto', 'auto']} />
              <ZAxis type="number" dataKey="seconds" range={[60, 260]} name="train s" />
              <Tooltip {...TOOLTIP} cursor={{ strokeDasharray: '3 3' }}
                formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : String(value), name]} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {archChart.map((entry, index) => (
                <Scatter key={entry.name} name={entry.name} data={[entry]}
                  fill={['#38bdf8', '#34d399', '#fbbf24', '#f472b6', '#a78bfa'][index % 5]} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
          <div className="mt-4">
            <Table rows={arch} columns={['experiment', 'architecture', 'encoder', 'miou', 'params', 'train_seconds']} />
          </div>
        </Card>
      )}

      {pretrain.length > 0 && (
        <Card title="Encoder initialisation" subtitle="ImageNet-pretrained encoder versus training from scratch at the same labelled-data budget.">
          <Table rows={pretrain} columns={['experiment', 'encoder_weights', 'seed', 'miou', 'epochs_ran']} />
        </Card>
      )}

      {compression.length > 0 && (
        <Card
          title="Compression"
          subtitle="FP32 versus INT8, both measured through ONNX Runtime on CPU with a pinned thread count — Apple silicon has no INT8 path on the GPU, so a same-backend comparison is the only honest one."
        >
          <Table rows={compression as Row[]}
            columns={['precision', 'miou', 'miou_delta', 'latency_ms_mean', 'latency_ms_p95', 'speedup', 'size_mb', 'size_reduction']} />
        </Card>
      )}
    </div>
  )
}
