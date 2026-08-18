import type { Row } from '../api'

/** Compact table for a results CSV, with numeric columns right-aligned and rounded. */
export function Table({
  rows,
  columns,
  highlight,
  precision = 4,
}: {
  rows: Row[]
  columns?: string[]
  /** Row index to emphasise, e.g. the winning configuration. */
  highlight?: number
  precision?: number
}) {
  if (!rows.length) return <p className="text-sm text-slate-500">No data yet.</p>
  const keys = columns ?? Object.keys(rows[0])

  const format = (value: Row[string]) => {
    if (value === null || value === undefined) return '—'
    if (typeof value === 'number') {
      if (Number.isInteger(value)) return value.toLocaleString()
      return value.toFixed(Math.abs(value) < 1 ? precision : 2)
    }
    return value
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wide text-slate-500">
            {keys.map((key) => (
              <th key={key} className="px-3 py-2 font-medium whitespace-nowrap">
                {key.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={index}
              className={`border-b border-slate-800/60 ${
                index === highlight ? 'bg-emerald-500/10 text-emerald-200' : 'text-slate-300'
              }`}
            >
              {keys.map((key) => (
                <td
                  key={key}
                  className={`px-3 py-2 whitespace-nowrap ${
                    typeof row[key] === 'number' ? 'text-right font-mono' : ''
                  }`}
                >
                  {format(row[key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
