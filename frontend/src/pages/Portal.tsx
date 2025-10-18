import { useEffect, useMemo, useState } from 'react'

type Device = { id: string; type: 'CPU'|'GPU'|'Edge'; name: string; status: string };
const envAny = (import.meta as unknown as { env?: Record<string,string> });
const API_BASE: string = envAny?.env?.VITE_BACKEND_URL || 'http://localhost:8787';

export default function Portal(){
  const [devices, setDevices] = useState<Device[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [result, setResult] = useState<{device:string; time:number}|null>(null);
  const [code, setCode] = useState<string>('');
  const [wsOk, setWsOk] = useState<boolean>(false);
  const [health, setHealth] = useState<string>('unknown');

  useEffect(()=>{
    let closed = false;
    let retryTimer: number | undefined;
    function connect() {
      try {
        const url = API_BASE.replace(/^http/,'ws');
        const ws = new WebSocket(url);
        ws.onopen = ()=> setWsOk(true);
        ws.onmessage = (ev)=>{
          try{
            const { line } = JSON.parse(String(ev.data));
            if (!closed && line) setLogs(prev=>[...prev, line].slice(-500));
          }catch(_e){ /* ignore */ }
        };
        ws.onclose = ()=> {
          setWsOk(false);
          if (!closed) {
            retryTimer = window.setTimeout(connect, 1500) as unknown as number;
          }
        };
        return ws;
      } catch (_e) {
        setWsOk(false);
        retryTimer = window.setTimeout(connect, 1500) as unknown as number;
        return null;
      }
    }
    const ws = connect();
    return ()=>{
      closed = true;
      try { if (ws) ws.close(); } catch(_e){ /* noop */ }
      if (retryTimer) window.clearTimeout(retryTimer);
    };
  },[]);

  async function connectDevices(){
    await fetch(`${API_BASE}/connect`, { method:'POST' });
    const r = await fetch(`${API_BASE}/devices`);
    const j = await r.json();
    setDevices(j.devices || []);
  }

  async function runDemo(){
    const r = await fetch(`${API_BASE}/run-demo`, { method:'POST' });
    const j = await r.json();
    if (j && j.result === 'success') setResult({ device: j.device, time: j.time });
  }

  async function genKernel(){
    const r = await fetch(`${API_BASE}/generate-kernel`, { method:'POST' });
    const j = await r.json();
    setCode(j.code || '');
  }

  async function checkHealth(){
    try{
      const r = await fetch(`${API_BASE}/health`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setHealth(j.ok ? 'ok' : 'bad');
    }catch(_e){ setHealth('down'); }
  }

  const deviceDots = useMemo(()=>{
    const colors: Record<string,string> = { CPU:'#6ee7b7', GPU:'#93c5fd', Edge:'#fca5a5' };
    return devices.map((d,i)=> (
      <g key={d.id} transform={`translate(${40+i*60},40)`}>
        <circle r={14} fill={colors[d.type]||'#fff'} opacity={d.status==='connected'?1:0.3} />
        <text x={0} y={28} textAnchor='middle' fill='#94a3b8' fontSize='10'>{d.type}</text>
      </g>
    ));
  },[devices]);

  useEffect(()=>{ document.title = 'U-HOP Demo Portal'; }, []);

  return (
    <div className='min-h-screen bg-[#0b1020] text-[#e5e7eb] font-mono'>
      <div className='max-w-6xl mx-auto p-6'>
        <header className='flex items-center justify-between mb-6'>
          <h1 className='text-2xl md:text-3xl font-bold tracking-wide'>U-HOP Demo Portal</h1>
          <div className='text-xs text-[#94a3b8]'>Sevenloops — U-HOP· <a className='underline' href='https://github.com/sevenloops/uhop' target='_blank' rel='noopener noreferrer'>GitHub</a></div>
        </header>

        <div className='grid grid-cols-1 md:grid-cols-3 gap-6'>
          {/* Controls */}
          <section className='md:col-span-1 space-y-3'>
            <button className='w-full py-2 rounded bg-[#111827] hover:bg-[#1f2937] border border-[#374151]' onClick={connectDevices}>Connect Devices</button>
            <button className='w-full py-2 rounded bg-[#111827] hover:bg-[#1f2937] border border-[#374151]' onClick={runDemo}>Run Matmul Test</button>
            <button className='w-full py-2 rounded bg-[#111827] hover:bg-[#1f2937] border border-[#374151]' onClick={genKernel}>Generate Kernel</button>
            <button className='w-full py-2 rounded bg-[#0b1324] hover:bg-[#111c34] border border-[#374151]' onClick={checkHealth}>Health Check</button>
            <div className='text-xs text-[#9ca3af]'>Backend: {API_BASE} · WS: {wsOk? 'ok':'off'} · Health: {health}</div>
            <svg width='100%' height='100' viewBox='0 0 320 100'>
              <rect x='0' y='0' width='100%' height='100%' fill='transparent' stroke='#334155' />
              {deviceDots}
            </svg>
          </section>

          {/* System Log Console */}
          <section className='md:col-span-2'>
            <div className='text-sm mb-2'>System Log Console</div>
            <div className='h-64 overflow-auto rounded border border-[#374151] bg-[#0f172a] p-3 text-xs whitespace-pre-wrap'>
              {logs.length===0 ? <div className='text-[#64748b]'>Awaiting events...</div> : logs.map((l, i)=> <div key={i}>{l}</div>)}
            </div>
          </section>

          {/* Result panel */}
          <section className='md:col-span-1'>
            <div className='text-sm mb-2'>Result</div>
            <div className='rounded border border-[#374151] bg-[#0f172a] p-3 text-xs'>
              {result ? (
                <div>
                  <div>Device: <span className='text-[#93c5fd]'>{result.device}</span></div>
                  <div>Time: <span className='text-[#6ee7b7]'>{result.time} ms</span></div>
                </div>
              ) : <div className='text-[#64748b]'>No result yet</div>}
            </div>
          </section>

          {/* Generated Code View */}
          <section className='md:col-span-2'>
            <div className='text-sm mb-2'>Generated Code</div>
            <pre className='rounded border border-[#374151] bg-[#0b1220] p-3 text-xs overflow-auto'><code>{code || '// Click "Generate Kernel" to view an example CUDA/OpenCL kernel'}</code></pre>
          </section>
        </div>
      </div>
    </div>
  )
}
