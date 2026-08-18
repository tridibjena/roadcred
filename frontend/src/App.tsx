import { useState } from 'react'
import { Layout, type PageKey } from './components/Layout'
import Inference from './pages/Inference'
import ModelComparison from './pages/ModelComparison'
import Robustness from './pages/Robustness'

export default function App() {
  const [page, setPage] = useState<PageKey>('inference')
  return (
    <Layout page={page} onNavigate={setPage}>
      {page === 'inference' && <Inference />}
      {page === 'comparison' && <ModelComparison />}
      {page === 'robustness' && <Robustness />}
    </Layout>
  )
}
