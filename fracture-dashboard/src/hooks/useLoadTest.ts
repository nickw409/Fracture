import { useState, useRef, useCallback } from 'react';
import { streamChat } from '../api/sse';

interface LoadTestStats {
  requestsSent: number;
  requestsCompleted: number;
  requestsFailed: number;
  totalTokens: number;
  running: boolean;
}

const PROMPTS = [
  'Explain how a CPU cache works in two sentences.',
  'What is the difference between TCP and UDP?',
  'Write a haiku about distributed systems.',
  'Describe the purpose of a load balancer.',
  'What is pipeline parallelism in machine learning?',
  'Explain what a mutex is to a five year old.',
  'List three benefits of using Rust for systems programming.',
  'What happens when you type a URL into a browser?',
];

export function useLoadTest() {
  const [stats, setStats] = useState<LoadTestStats>({
    requestsSent: 0,
    requestsCompleted: 0,
    requestsFailed: 0,
    totalTokens: 0,
    running: false,
  });
  const abortRef = useRef<AbortController | null>(null);
  const runningRef = useRef(false);

  const start = useCallback((maxTokens = 64) => {
    if (runningRef.current) return;
    runningRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;

    setStats({
      requestsSent: 0,
      requestsCompleted: 0,
      requestsFailed: 0,
      totalTokens: 0,
      running: true,
    });

    const fireNext = () => {
      if (!runningRef.current) return;

      const prompt = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
      setStats((s) => ({ ...s, requestsSent: s.requestsSent + 1 }));

      let tokens = 0;
      streamChat(
        {
          messages: [{ role: 'user', content: prompt }],
          max_tokens: maxTokens,
          temperature: 0.7,
        },
        () => {
          tokens++;
        },
        () => {
          setStats((s) => ({
            ...s,
            requestsCompleted: s.requestsCompleted + 1,
            totalTokens: s.totalTokens + tokens,
          }));
          // Fire the next request after this one completes.
          fireNext();
        },
        () => {
          setStats((s) => ({
            ...s,
            requestsFailed: s.requestsFailed + 1,
          }));
          // Continue even after failures.
          fireNext();
        },
        controller.signal,
      );
    };

    fireNext();
  }, []);

  const stop = useCallback(() => {
    runningRef.current = false;
    abortRef.current?.abort();
    abortRef.current = null;
    setStats((s) => ({ ...s, running: false }));
  }, []);

  return { stats, start, stop };
}
