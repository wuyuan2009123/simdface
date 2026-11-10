export async function analyzeServer(mode, first, second, options, signal) {
  const data = new FormData();
  data.append('first', first);
  if (second) data.append('second', second);
  const query = new URLSearchParams({ score: options.score, backend: options.backend });
  if ((mode === 'compare' || mode === 'vectors') && options.threshold !== '') query.set('threshold', options.threshold);
  const response = await fetch('/api/' + mode + '?' + query, {
    method: 'POST', body: data, headers: { 'X-SimdFace-Request': '1' }, signal
  });
  const result = await response.json().catch(() => ({}));
  return { result, status: response.status, ok: response.ok };
}
