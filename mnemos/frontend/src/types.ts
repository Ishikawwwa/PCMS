export type Node = { id: string; type: string; title: string; level: number; status: string; version: number; created_by_event: string; proposition?: string; content?: string; vector?: number[]; embedding_model_id?: string; conditions?: string[]; exceptions?: string[]; [key: string]: unknown };
export type Hyperedge = { id: string; type: 'hyperedge'; member_ids: string[]; status: string; cohesion: number; min_pairwise_cosine: number; mean_pairwise_cosine: number; level: number; [key: string]: unknown };
export type Graph = { schema_version?:string; nodes: Node[]; links: { id: string; source: string; target: string; type: string }[]; hyperedges: Hyperedge[]; graph_version: number; projection: { coordinates: Record<string, number[]>; explained_variance: number; id: string } | null };
export type Run = { id: string; name: string; settings: Record<string, any>; version: number; status: string; graph: Graph };
export type Event = { id: string; seq: number; kind: string; created_at: string; data: Record<string, any> };
export type Job = { id: string; status: string; error: string | null; result: Record<string, any> | null };
export type Inspection = { object: Node | Hyperedge; sources: Node[]; source_ids: string[]; neighbors: {id: string;title: string;cosine: number}[]; impact: {node_ids: string[];hyperedge_ids: string[]}; generating_event: Event; model_calls: unknown[] };
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch('/api' + path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail || response.statusText)); }
  return response.json();
}
