import { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { HEALTH_POLL_MS } from '../../lib/constants';

export default function StatusBar() {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        await api.health();
        if (active) setConnected(true);
      } catch {
        if (active) setConnected(false);
      }
    };
    check();
    const id = setInterval(check, HEALTH_POLL_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="flex items-center h-7 px-3 bg-gray-900 border-t border-gray-700 text-xs text-gray-500 gap-2">
      <span
        className={`inline-block w-2 h-2 rounded-full ${
          connected ? 'bg-emerald-400' : 'bg-red-400'
        }`}
      />
      <span>{connected ? 'Connected' : 'Disconnected'}</span>
      <span className="ml-auto">Fracture Dashboard</span>
    </div>
  );
}
