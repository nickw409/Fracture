import type { ChatMessage } from '../../api/types';

interface Props {
  message: ChatMessage;
  isStreaming?: boolean;
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-3 text-sm whitespace-pre-wrap ${
          isUser ? 'bg-blue-400/10 text-gray-100' : 'bg-gray-800 text-gray-100'
        }`}
      >
        {message.content}
        {isStreaming && (
          <span className="inline-block w-1.5 h-4 ml-0.5 bg-gray-400 animate-pulse" />
        )}
      </div>
    </div>
  );
}
