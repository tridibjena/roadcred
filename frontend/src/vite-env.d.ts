/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Override the API base URL; defaults to the dev proxy at /api. */
  readonly VITE_API_BASE?: string
}
interface ImportMeta {
  readonly env: ImportMetaEnv
}
