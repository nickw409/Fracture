import { NavLink } from 'react-router-dom';
import { Activity, MessageSquare, Layers } from 'lucide-react';

const links = [
  { to: '/', icon: Activity, label: 'Cluster' },
  { to: '/playground', icon: MessageSquare, label: 'Playground' },
  { to: '/scheduler', icon: Layers, label: 'Scheduler' },
];

export default function Sidebar() {
  return (
    <nav className="flex flex-col w-16 bg-gray-900 border-r border-gray-700 items-center py-4 gap-2">
      <div className="text-sm font-bold text-gray-300 mb-4 tracking-wider">F</div>
      {links.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          title={label}
          className={({ isActive }) =>
            `flex items-center justify-center w-10 h-10 rounded-lg transition-colors duration-150 ${
              isActive
                ? 'bg-gray-800 text-emerald-400'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
            }`
          }
        >
          <Icon size={20} />
        </NavLink>
      ))}
    </nav>
  );
}
