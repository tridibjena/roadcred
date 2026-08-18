/** Typed client for the RoadSense FastAPI backend. */

// Vite proxies /api to the backend in development (see vite.config.ts), so the browser
// always sees a same-origin request and no CORS preflight is needed.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export interface ClassShare {
  name: string
  index: number
  pixel_fraction: number
  mean_confidence: number
  colour: [number, number, number]
}

export interface PredictResponse {
  width: number
  height: number
  class_names: string[]
  classes: ClassShare[]
  mean_confidence: number
  low_confidence_fraction: number
  temperature: number
  calibrated: boolean
  inference_ms: number
  mask_png: string
  overlay_png: string
  confidence_png: string
}

export interface ModelInfo {
  loaded: boolean
  architecture: string | null
  encoder: string | null
  class_names: string[]
  imgsz: number[]
  precision: string | null
  val_miou: number | null
  temperature: number | null
  model_path: string | null
}

export type Row = Record<string, string | number | null>

export interface ResultsPayload {
  tables: Record<string, Row[]>
  reports: Record<string, unknown>
  figures: string[]
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText)
    throw new Error(`${response.status}: ${detail.slice(0, 300)}`)
  }
  return response.json() as Promise<T>
}

export const getHealth = () => request<{ status: string; model: ModelInfo }>('/health')
export const getResults = () => request<ResultsPayload>('/results')

export function predict(file: File): Promise<PredictResponse> {
  const body = new FormData()
  body.append('image', file)
  return request<PredictResponse>('/predict', { method: 'POST', body })
}

export const figureUrl = (name: string) => `${BASE}/figures/${name}`
