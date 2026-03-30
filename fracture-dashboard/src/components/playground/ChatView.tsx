import { useRef, useEffect, useState, type KeyboardEvent } from 'react';
import type { ChatMessage } from '../../api/types';
import MessageBubble from './MessageBubble';
import { Send, Square } from 'lucide-react';

interface Props {
  messages: ChatMessage[];
  isGenerating: boolean;
  onSend: (content: string) => void;
  onAbort: () => void;
}

export default function ChatView({ messages, isGenerating, onSend, onAbort }: Props) {
  const [input, setInput] = useState('');
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isGenerating) return;
    setInput('');
    onSend(text);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto space-y-3 pb-4">
        {messages.length === 0 && (
          <div className="text-gray-500 text-sm text-center mt-20">
            Send a message to start a conversation.
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            message={msg}
            isStreaming={isGenerating && i === messages.length - 1 && msg.role === 'assistant'}
          />
        ))}
        <div ref={endRef} />
      </div>

      <div className="flex gap-2 pt-3 border-t border-gray-800">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          rows={1}
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-500 resize-none focus:outline-none focus:border-gray-600"
        />
        {isGenerating ? (
          <button
            onClick={onAbort}
            className="flex items-center justify-center w-10 h-10 bg-red-400/10 text-red-400 rounded-lg hover:bg-red-400/20 transition-colors duration-150"
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="flex items-center justify-center w-10 h-10 bg-blue-400/10 text-blue-400 rounded-lg hover:bg-blue-400/20 transition-colors duration-150 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <Send size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
