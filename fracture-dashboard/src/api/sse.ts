import type { ChatCompletionChunk, ChatCompletionRequest } from './types';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export async function streamChat(
  request: ChatCompletionRequest,
  onChunk: (chunk: ChatCompletionChunk) => void,
  onDone: () => void,
  onError: (error: Error) => void,
  signal?: AbortSignal,
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...request, stream: true }),
      signal,
    });

    if (!res.ok || !res.body) {
      throw new Error(`Stream error: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed === '') continue;
        if (trimmed === 'data: [DONE]') {
          onDone();
          return;
        }
        if (trimmed.startsWith('data: ')) {
          const json = trimmed.slice(6);
          try {
            const chunk: ChatCompletionChunk = JSON.parse(json);
            onChunk(chunk);
          } catch {
            // Skip malformed chunks
          }
        }
      }
    }
    onDone();
  } catch (err) {
    if (signal?.aborted) return;
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}
