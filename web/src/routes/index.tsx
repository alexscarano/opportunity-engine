import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'

type Health = { status: string; db: string }

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  const health = useQuery({
    queryKey: ['health'],
    queryFn: async (): Promise<Health> => {
      const r = await fetch('/api/health')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json()
    },
  })

  if (health.isPending) return <p>Carregando…</p>
  if (health.isError) return <p>API fora do ar: {health.error.message}</p>
  return (
    <p>
      API: {health.data.status} · banco: {health.data.db}
    </p>
  )
}
