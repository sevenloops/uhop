import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Docs from './pages/Docs';
import About from './pages/About';
import NotFound from './pages/NotFound';
import Portal from './pages/Portal';
import { useEffect } from 'react';

function SessionRedirect() {
  const navigate = useNavigate();
  useEffect(() => {
    const r = sessionStorage.getItem('redirect');
    if (r) {
      sessionStorage.removeItem('redirect');
      navigate(r, { replace: true });
    }
  }, [navigate]);
  return null;
}

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <div className="flex flex-col min-h-screen w-full relative bg-background text-foreground">
        <SessionRedirect />
        <Navbar />
        <main className="flex-1 pt-16">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/docs" element={<Docs />} />
            <Route path="/demo" element={<Portal />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
