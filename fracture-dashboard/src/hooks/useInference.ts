import { useState, useRef, useCallback } from 'react';
import type { ChatMessage, ChatCompletionChunk } from '../api/types';
import { streamChat } from '../api/sse';

export interface GenerationStats {
  timeToFirstTokenMs: number | null;
  tokensPerSecond: number | null;
  promptTokens: number | null;
  completionTokens: number;
  totalTokens: number | null;
  finishReason: string | null;
}

interface InferenceParams {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  seed: string;
}

export function useInference() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [stats, setStats] = useState<GenerationStats>({
    timeToFirstTokenMs: null,
    tokensPerSecond: null,
    promptTokens: null,
    completionTokens: 0,
    totalTokens: null,
    finishReason: null,
  });
  const abortRef = useRef<AbortController | null>(null);
  const startTimeRef = useRef<number>(0);
  const firstTokenTimeRef = useRef<number | null>(null);
  const tokenCountRef = useRef(0);

  const send = useCallback(
    async (content: string, params: InferenceParams) => {
      const userMsg: ChatMessage = { role: 'user', content };
      const newMessages = [...messages, userMsg];
      setMessages([...newMessages, { role: 'assistant', content: '' }]);
      setIsGenerating(true);
      setStats({
        timeToFirstTokenMs: null,
        tokensPerSecond: null,
        promptTokens: null,
        completionTokens: 0,
        totalTokens: null,
        finishReason: null,
      });

      const controller = new AbortController();
      abortRef.current = controller;
      startTimeRef.current = performance.now();
      firstTokenTimeRef.current = null;
      tokenCountRef.current = 0;

      await streamChat(
        {
          messages: newMessages,
          temperature: params.temperature,
          top_p: params.topP,
          top_k: params.topK,
          max_tokens: params.maxTokens,
          seed: params.seed ? Number(params.seed) : undefined,
        },
        (chunk: ChatCompletionChunk) => {
          const delta = chunk.choices[0]?.delta?.content ?? '';
          if (delta) {
            if (firstTokenTimeRef.current === null) {
              firstTokenTimeRef.current = performance.now();
              setStats((s) => ({
                ...s,
                timeToFirstTokenMs: firstTokenTimeRef.current! - startTimeRef.current,
              }));
            }
            tokenCountRef.current++;
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = { ...last, content: last.content + delta };
              return updated;
            });
            const elapsed = (performance.now() - (firstTokenTimeRef.current ?? startTimeRef.current)) / 1000;
            setStats((s) => ({
              ...s,
              completionTokens: tokenCountRef.current,
              tokensPerSecond: elapsed > 0 ? tokenCountRef.current / elapsed : null,
            }));
          }

          if (chunk.choices[0]?.finish_reason) {
            setStats((s) => ({
              ...s,
              finishReason: chunk.choices[0].finish_reason,
              promptTokens: chunk.usage?.prompt_tokens ?? null,
              totalTokens: chunk.usage?.total_tokens ?? null,
            }));
          }
        },
        () => {
          setIsGenerating(false);
          abortRef.current = null;
        },
        () => {
          setIsGenerating(false);
          abortRef.current = null;
        },
        controller.signal,
      );
    },
    [messages],
  );

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    setStats({
      timeToFirstTokenMs: null,
      tokensPerSecond: null,
      promptTokens: null,
      completionTokens: 0,
      totalTokens: null,
      finishReason: null,
    });
  }, []);

  return { messages, isGenerating, stats, send, abort, clear };
}
