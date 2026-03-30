import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Shell from './components/layout/Shell';
import ClusterPage from './pages/ClusterPage';
import PlaygroundPage from './pages/PlaygroundPage';
import SchedulerPage from './pages/SchedulerPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<ClusterPage />} />
          <Route path="playground" element={<PlaygroundPage />} />
          <Route path="scheduler" element={<SchedulerPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
