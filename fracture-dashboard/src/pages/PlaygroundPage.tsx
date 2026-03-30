import { useState } from 'react';
import { useInference } from '../hooks/useInference';
import ChatView from '../components/playground/ChatView';
import ParamControls from '../components/playground/ParamControls';
import GenerationStats from '../components/playground/GenerationStats';
import { Trash2 } from 'lucide-react';

export default function PlaygroundPage() {
  const { messages, isGenerating, stats, send, abort, clear } = useInference();
  const [params, setParams] = useState({
    temperature: 0.7,
    topP: 1.0,
    topK: 0,
    maxTokens: 256,
    seed: '',
  });

  return (
    <div className="flex gap-6 h-full">
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-semibold">Inference Playground</h1>
          {messages.length > 0 && (
            <button
              onClick={clear}
              disabled={isGenerating}
              className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200 transition-colors duration-150 disabled:opacity-30"
            >
              <Trash2 size={14} /> Clear
            </button>
          )}
        </div>
        <ChatView
          messages={messages}
          isGenerating={isGenerating}
          onSend={(content) => send(content, params)}
          onAbort={abort}
        />
      </div>

      <div className="w-72 flex-shrink-0 space-y-4">
        <ParamControls params={params} onChange={setParams} />
        <GenerationStats stats={stats} />
      </div>
    </div>
  );
}
