import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.dirname(__dirname);

const app = express();
const PORT = process.env.PORT || 8787;
const ORIGIN = process.env.CORS_ORIGIN || '*';

app.use(cors({ origin: ORIGIN }));
app.use(express.json());

// State for device information
let deviceInfo = null;
let isConnected = false;

// WebSocket server for logs
let server;
try {
  server = app.listen(PORT, '::', () => {
    const addr = server.address();
    const where = typeof addr === 'string' ? addr : `${addr.address}:${addr.port}`;
    console.log(`[backend] listening on ${where}`);
  });
} catch (e) {
  // Fallback to IPv4 only
  server = app.listen(PORT, '0.0.0.0', () => {
    const addr = server.address();
    const where = typeof addr === 'string' ? addr : `${addr.address}:${addr.port}`;
    console.log(`[backend] listening on ${where}`);
  });
}
const wss = new WebSocketServer({ server });

// Agent management
const AGENT_TOKEN = process.env.AGENT_TOKEN || null;
let agentSocket = null; // single active agent for now
const pending = new Map(); // id -> {resolve, reject, timer}
function makeId() { return Math.random().toString(36).slice(2) + Date.now().toString(36); }
function isOpen(s) { return !!(s && s.readyState === 1); }

function wsSend(ws, obj) {
  try { ws.send(JSON.stringify(obj)); } catch (e) {}
}

function askAgent(action, params = {}, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    if (!isOpen(agentSocket)) {
      return reject(new Error('no_agent'));
    }
    const id = makeId();
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error('agent_timeout'));
    }, timeoutMs);
    pending.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
    wsSend(agentSocket, { id, type: 'request', action, params });
  });
}

wss.on('connection', (ws, req) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const isAgent = url.pathname === '/agent';
  console.log(`[backend][ws] connection path=${url.pathname}`);

  if (isAgent) {
    // Only keep one agent for simplicity
    if (agentSocket && agentSocket.readyState === agentSocket.OPEN) {
      try { agentSocket.close(); } catch (e) {}
    }
    agentSocket = ws;
    broadcast('[backend] agent connected');

    ws.on('message', (data) => {
      let msg = null;
      try { msg = JSON.parse(String(data)); } catch { return; }
      if (!msg) return;
      if (msg.type === 'hello') {
        const token = msg.token || null;
        if (AGENT_TOKEN && token !== AGENT_TOKEN) {
          broadcast('[backend] agent token mismatch; disconnecting');
          try { ws.close(); } catch (e) {}
          return;
        }
        broadcast(`[agent] hello ${msg.agent || ''} v${msg.version || ''}`);
        return;
      }
      if (msg.type === 'response' && msg.id && pending.has(msg.id)) {
        const { resolve, reject } = pending.get(msg.id);
        pending.delete(msg.id);
        if (msg.ok) resolve(msg.data); else reject(new Error(msg.error || 'agent_error'));
      } else if (msg.type === 'log' && msg.line) {
        broadcast(`[agent] ${msg.line}`);
      }
    });
    ws.on('close', () => {
      if (agentSocket === ws) agentSocket = null;
      broadcast('[backend] agent disconnected');
    });
    return; // do not set generic handlers for agent ws
  }

  // Non-agent (browser) websocket for logs
  ws.on('message', () => {});
});

function broadcast(line) {
  const payload = JSON.stringify({ t: Date.now(), line });
  for (const client of wss.clients) {
    try { client.send(payload); } catch (e) {}
  }
}

// Helper to run UHOP CLI commands
function runUhopCommand(args, options = {}) {
  return new Promise((resolve, reject) => {
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    const child = spawn(pythonPath, ['-m', 'uhop.cli', ...args], {
      cwd: projectRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
      ...options
    });
    
    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout: stdout.trim(), stderr: stderr.trim(), code });
      } else {
        reject(new Error(`Command failed with code ${code}: ${stderr}`));
      }
    });
    
    child.on('error', (error) => {
      reject(error);
    });
  });
}

// Helper to run UHOP web API
function runUhopDemo(size = 128, iters = 2) {
  return new Promise((resolve, reject) => {
    const pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    const projectRootForPython = projectRoot.replace(/\\/g, '/');
    const child = spawn(pythonPath, ['-c', `
import sys
sys.path.insert(0, r'${projectRoot}')
from uhop.web_api import _demo_matmul
import json
try:
    result = _demo_matmul(${size}, ${iters})
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
`], {
      cwd: projectRoot,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let stdout = '';
    let stderr = '';
    
    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    child.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdout.trim());
          resolve(result);
        } catch (e) {
          reject(new Error(`Failed to parse result: ${e.message}`));
        }
      } else {
        reject(new Error(`Demo failed with code ${code}: ${stderr}`));
      }
    });
    
    child.on('error', (error) => {
      reject(error);
    });
  });
}

// Connect devices (real hardware detection)
app.post('/connect', async (req, res) => {
  try {
    broadcast('[U-HOP] Detecting hardware and initializing backends...');
    
    // Prefer local agent if connected
    let info = null;
    try {
      info = await askAgent('info', {}, 10000);
      broadcast('[U-HOP] Using local agent hardware info');
    } catch (_e) {
      // Fallback to server-side detection
      const infoResult = await runUhopCommand(['info', '--json']);
      info = JSON.parse(infoResult.stdout);
      broadcast('[U-HOP] Using server hardware info');
    }
    deviceInfo = info;
    
    // Convert to device format expected by frontend
    const devices = [];
    
    // Add detected device with better priority
    let deviceType = 'CPU';
    let deviceName = deviceInfo.name || 'Unknown';
    
    // Priority: CUDA GPU > OpenCL GPU > CPU
    if (deviceInfo.kind === 'cuda') {
      deviceType = 'GPU';
      deviceName = deviceInfo.name || 'NVIDIA GPU';
    } else if (deviceInfo.kind === 'opencl') {
      deviceType = 'GPU';
      deviceName = deviceInfo.name || `${deviceInfo.vendor} GPU`;
    } else if (deviceInfo.kind === 'opencl-cpu') {
      // Only fall back to CPU if no GPU OpenCL
      if (deviceInfo.opencl_available) {
        deviceType = 'CPU';
        deviceName = deviceInfo.name || `${deviceInfo.vendor} CPU via OpenCL`;
      } else {
        deviceType = 'CPU';
        deviceName = deviceInfo.name || 'CPU';
      }
    }
    
    devices.push({
      id: 'primary-device',
      type: deviceType,
      name: deviceName,
      status: 'connected'
    });
    
    // Add additional info if available
    if (deviceInfo.torch_available) {
      devices.push({
        id: 'torch-backend',
        type: 'Edge',
        name: 'PyTorch Backend',
        status: 'connected'
      });
    }
    
    isConnected = true;
    broadcast(`[U-HOP] Connected to ${deviceType}: ${deviceName}`);
    broadcast(`[U-HOP] Backend capabilities: Torch=${deviceInfo.torch_available}, OpenCL=${deviceInfo.opencl_available}`);
    
    res.json({ ok: true, devices });
  } catch (error) {
    broadcast(`[U-HOP] Error during hardware detection: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// Device status
app.get('/devices', (req, res) => {
  if (!isConnected || !deviceInfo) {
    res.json({ devices: [] });
    return;
  }
  
  const devices = [];
  
  let deviceType = 'CPU';
  let deviceName = deviceInfo.name || 'Unknown';
  
  // Priority: CUDA GPU > OpenCL GPU > CPU
  if (deviceInfo.kind === 'cuda') {
    deviceType = 'GPU';
    deviceName = deviceInfo.name || 'NVIDIA GPU';
  } else if (deviceInfo.kind === 'opencl') {
    deviceType = 'GPU';
    deviceName = deviceInfo.name || `${deviceInfo.vendor} GPU`;
  } else if (deviceInfo.kind === 'opencl-cpu') {
    if (deviceInfo.opencl_available) {
      deviceType = 'CPU';
      deviceName = deviceInfo.name || `${deviceInfo.vendor} CPU via OpenCL`;
    } else {
      deviceType = 'CPU';
      deviceName = deviceInfo.name || 'CPU';
    }
  }
  
  devices.push({
    id: 'primary-device',
    type: deviceType,
    name: deviceName,
    status: 'connected'
  });
  
  res.json({ devices });
});

// Run demo (real UHOP matmul)
app.post('/run-demo', async (req, res) => {
  try {
    if (!isConnected) {
      broadcast('[U-HOP] No devices connected. Run "Connect Devices" first.');
      return res.status(400).json({ error: 'no_devices' });
    }
    
    const size = parseInt(req.body.size) || 128;
    const iters = parseInt(req.body.iters) || 2;
    
    broadcast(`[U-HOP] Running matmul benchmark (${size}x${size}, ${iters} iterations)...`);
    
    let result = null;
    try {
      // Try local agent first
      result = await askAgent('run_demo', { size, iters }, 20000);
      broadcast('[U-HOP] Demo executed on local agent');
    } catch (_e) {
      // Fallback to server
      result = await runUhopDemo(size, iters);
      broadcast('[U-HOP] Demo executed on server');
    }
    
    // Parse the stdout to get individual lines for broadcasting
    const lines = result.stdout.split('\n').filter(line => line.trim());
    lines.forEach(line => broadcast(`[U-HOP] ${line}`));
    
  const deviceType = deviceInfo?.kind === 'cuda' ? 'GPU' : 
            deviceInfo?.kind === 'opencl' ? 'GPU' : 'CPU';
    
    res.json({ 
      result: 'success', 
      time: (result.timings.uhop * 1000).toFixed(2), // Convert to ms
      device: deviceType,
      timings: result.timings
    });
  } catch (error) {
    broadcast(`[U-HOP] Demo failed: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// Generate kernel (real AI generation)
app.post('/generate-kernel', async (req, res) => {
  try {
    broadcast('[U-HOP][AI] Generating optimized kernel using AI...');
    
    // Prefer local agent for generation (uses user's API key if set locally)
    try {
      const data = await askAgent('generate_kernel', { target: 'opencl' }, 60000);
      broadcast('[U-HOP][AI] Kernel generated via agent');
      return res.json(data);
    } catch (_e) {
      // Fallback to server behavior
      if (!process.env.OPENAI_API_KEY) {
        broadcast('[U-HOP][AI] Warning: OPENAI_API_KEY not set on server, using fallback...');
        const fallbackCode = `// Generated OpenCL kernel (fallback - set OPENAI_API_KEY for real AI generation)\n__kernel void generated_matmul(\n    const int M, const int N, const int K,\n    __global const float* A,\n    __global const float* B, \n    __global float* C\n) {\n    int row = get_global_id(0);\n    int col = get_global_id(1);\n    \n    if (row >= M || col >= N) return;\n    \n    float acc = 0.0f;\n    for (int k = 0; k < K; ++k) {\n        acc += A[row * K + k] * B[k * N + col];\n    }\n    C[row * N + col] = acc;\n}`;
        return res.json({ language: 'opencl', code: fallbackCode });
      }
      const generateResult = await runUhopCommand(['ai-generate', 'matmul', '--target', 'opencl']);
      broadcast('[U-HOP][AI] Kernel generated on server');
      const fs = await import('fs');
      const generatedDir = path.join(projectRoot, 'uhop', 'generated_kernels');
      try {
        const files = fs.readdirSync(generatedDir)
          .filter(f => f.startsWith('ai_matmul_') && f.endsWith('.cl'))
          .map(f => ({
            name: f,
            path: path.join(generatedDir, f),
            mtime: fs.statSync(path.join(generatedDir, f)).mtime
          }))
          .sort((a, b) => b.mtime - a.mtime);
        if (files.length > 0) {
          const code = fs.readFileSync(files[0].path, 'utf8');
          return res.json({ language: 'opencl', code });
        } else {
          throw new Error('No generated files found');
        }
      } catch (readError) {
        broadcast(`[U-HOP][AI] Could not read generated file: ${readError.message}`);
        return res.status(500).json({ error: readError.message });
      }
    }
  } catch (error) {
    broadcast(`[U-HOP][AI] Generation failed: ${error.message}`);
    res.status(500).json({ error: error.message });
  }
});

// Health (real backend health)
app.get('/health', async (req, res) => {
  try {
    // Health: server UHOP info + agent status
    let serverOk = true;
    try { await runUhopCommand(['info', '--json']); }
    catch { serverOk = false; }
    const agentOk = isOpen(agentSocket);
    res.json({ ok: serverOk || agentOk, serverOk, agentOk });
  } catch (error) {
    res.status(500).json({ ok: false, error: error.message });
  }
});

// Extra endpoint: agent status
app.get('/agent-status', (req, res) => {
  const agentOk = isOpen(agentSocket);
  res.json({ connected: agentOk });
});
