import express from 'express';
import cors from 'cors';
import { WebSocketServer } from 'ws';

const app = express();
const PORT = process.env.PORT || 8787;
const ORIGIN = process.env.CORS_ORIGIN || '*';

app.use(cors({ origin: ORIGIN }));
app.use(express.json());

// In-memory device list (simulated)
const devices = [
  { id: 'cpu-1', type: 'CPU', name: 'Intel i7', status: 'disconnected' },
  { id: 'gpu-1', type: 'GPU', name: 'NVIDIA RTX', status: 'disconnected' },
  { id: 'edge-1', type: 'Edge', name: 'Raspberry Pi 5', status: 'disconnected' }
];

// WebSocket server for logs
const server = app.listen(PORT, '0.0.0.0', () => {
  const addr = server.address();
  const where = typeof addr === 'string' ? addr : `${addr.address}:${addr.port}`;
  console.log(`[backend] listening on ${where}`);
});
const wss = new WebSocketServer({ server });

function broadcast(line) {
  const payload = JSON.stringify({ t: Date.now(), line });
  for (const client of wss.clients) {
    try { client.send(payload); } catch (e) {}
  }
}

// Helper to wait
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Connect devices (simulate)
app.post('/connect', async (req, res) => {
  broadcast('[U-HOP] Initializing distributed fabric...');
  await sleep(300);
  for (const d of devices) {
    d.status = 'connected';
  }
  broadcast(`[U-HOP] Connected devices: ${devices.length}`);
  res.json({ ok: true, devices });
});

// Device status
app.get('/devices', (req, res) => {
  res.json({ devices });
});

// Run demo
app.post('/run-demo', async (req, res) => {
  const active = devices.filter((d) => d.status === 'connected');
  if (!active.length) {
    broadcast('[U-HOP] No devices connected.');
    return res.status(400).json({ error: 'no_devices' });
  }
  const gpu = active.find((d) => d.type === 'GPU') || active[0];
  broadcast(`[U-HOP] Dispatching matmul to ${gpu.type} node`);
  await sleep(250);
  broadcast('[U-HOP] Optimization kernel loaded: CUDA_v0.2');
  await sleep(250);
  const time = 8.23; // deterministic
  broadcast(`[U-HOP] Result: 1024x1024 matmul done in ${time} ms`);
  res.json({ result: 'success', time, device: gpu.type });
});

// Generate kernel (mock)
app.post('/generate-kernel', async (req, res) => {
  broadcast('[U-HOP][AI] Generating optimized kernel candidate...');
  await sleep(300);
  const samples = [
`// OpenCL kernel (mock)
__kernel void matmul(__global float* A, __global float* B, __global float* C, int N){
  int row = get_global_id(0);
  int col = get_global_id(1);
  float acc = 0.0f;
  for (int k = 0; k < N; ++k) acc += A[row*N+k] * B[k*N+col];
  C[row*N+col] = acc;
}`,
`// CUDA kernel (mock)
extern "C" __global__ void matmul(const float* A, const float* B, float* C, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k < N; ++k) acc += A[row*N+k] * B[k*N+col];
  C[row*N+col] = acc;
}`
  ];
  const code = samples[Date.now() % 2];
  broadcast('[U-HOP][AI] Candidate ready. Running smoke checks...');
  await sleep(200);
  broadcast('[U-HOP][AI] Validation passed.');
  res.json({ language: code.includes('__kernel') ? 'opencl' : 'cuda', code });
});

// Health
app.get('/health', (req, res) => res.json({ ok: true }));
