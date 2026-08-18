import { useEffect, useRef, useState } from 'react'
import { Card, Message, Stat } from '../components/Layout'
import { getHealth, predict, type ModelInfo, type PredictResponse } from '../api'

type LayerKey = 'original' | 'overlay' | 'mask' | 'confidence'

const LAYERS: { key: LayerKey; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'overlay', label: 'Overlay' },
  { key: 'mask', label: 'Mask' },
  { key: 'confidence', label: 'Confidence' },
]

export default function Inference() {
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [layer, setLayer] = useState<LayerKey>('overlay')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getHealth()
      .then((h) => setModel(h.model))
      .catch((e: Error) => setError(e.message))
  }, [])

  // Revoke the object URL when it is replaced, so repeated uploads do not leak blobs.
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  async function handleFile(file: File) {
    setBusy(true)
    setError(null)
    setResult(null)
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(file) })
    try {
      setResult(await predict(file))
      setLayer('overlay')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const source =
    layer === 'original'
      ? preview
      : result
        ? `data:image/png;base64,${
            layer === 'overlay' ? result.overlay_png
            : layer === 'mask' ? result.mask_png
            : result.confidence_png
          }`
        : null

  return (
    <div className="space-y-6">
      {error && <Message kind="error">{error}</Message>}
      {model && !model.loaded && (
        <Message kind="info">
          No model is loaded. Export one first:{' '}
          <code className="font-mono text-xs">
            python -m compression.quantize --checkpoint checkpoints/&lt;best&gt;.pt
          </code>
        </Message>
      )}

      <Card
        title="Segment an image"
        subtitle="Upload a road scene. The model predicts seven classes and returns a temperature-calibrated confidence map."
      >
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            const file = e.dataTransfer.files?.[0]
            if (file) void handleFile(file)
          }}
          onClick={() => inputRef.current?.click()}
          className="cursor-pointer rounded-lg border-2 border-dashed border-slate-700 px-6 py-10 text-center transition hover:border-sky-500/60 hover:bg-slate-900/60"
        >
          <p className="text-sm text-slate-300">Drop an image here, or click to choose one</p>
          <p className="mt-1 text-xs text-slate-500">JPEG or PNG · any resolution</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void handleFile(f) }}
          />
        </div>
        {busy && <p className="mt-3 text-sm text-sky-300">Running inference…</p>}
      </Card>

      {result && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat
              label="Mean confidence"
              value={`${(result.mean_confidence * 100).toFixed(1)}%`}
              hint={result.calibrated ? `calibrated, T = ${result.temperature.toFixed(3)}` : 'uncalibrated'}
            />
            <Stat
              label="Low confidence"
              value={`${(result.low_confidence_fraction * 100).toFixed(1)}%`}
              hint="pixels below 0.60"
            />
            <Stat label="Inference" value={`${result.inference_ms.toFixed(0)} ms`} hint="ONNX Runtime, CPU" />
            <Stat label="Resolution" value={`${result.width}×${result.height}`} />
          </div>

          <Card title="Prediction">
            <div className="mb-4 flex flex-wrap gap-1">
              {LAYERS.map((option) => (
                <button
                  key={option.key}
                  onClick={() => setLayer(option.key)}
                  disabled={option.key === 'original' && !preview}
                  className={`rounded-md px-3 py-1.5 text-xs transition disabled:opacity-40 ${
                    layer === option.key
                      ? 'bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/40'
                      : 'bg-slate-800/60 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
            {source && (
              <img
                src={source}
                alt={layer}
                className="w-full rounded-lg border border-slate-800 bg-slate-950"
              />
            )}
          </Card>

          <Card title="Class composition" subtitle="Share of predicted pixels and mean confidence per class.">
            <div className="space-y-2">
              {[...result.classes]
                .sort((a, b) => b.pixel_fraction - a.pixel_fraction)
                .map((entry) => (
                  <div key={entry.name} className="flex items-center gap-3">
                    <span
                      className="h-3 w-3 shrink-0 rounded-sm"
                      style={{ backgroundColor: `rgb(${entry.colour.join(',')})` }}
                    />
                    <span className="w-52 shrink-0 truncate text-xs text-slate-300">{entry.name}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(entry.pixel_fraction * 100, 0.4)}%`,
                          backgroundColor: `rgb(${entry.colour.join(',')})`,
                        }}
                      />
                    </div>
                    <span className="w-14 shrink-0 text-right font-mono text-xs text-slate-400">
                      {(entry.pixel_fraction * 100).toFixed(1)}%
                    </span>
                    <span className="w-14 shrink-0 text-right font-mono text-xs text-slate-500">
                      {entry.pixel_fraction > 0 ? `${(entry.mean_confidence * 100).toFixed(0)}%` : '—'}
                    </span>
                  </div>
                ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
