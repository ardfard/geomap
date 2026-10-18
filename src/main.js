import { HEIGHT_WGSL, RENDER_WGSL } from './shaders.js';
import { UNITS } from './geology.js';

const params = new URLSearchParams(location.search);
const GRID = Math.min(1024, Math.max(64, +params.get('grid') || 512)); // heightmap resolution
const WALL_SEGMENTS = GRID * 2; // segments per block side
const SHADOW_SIZE = 2048;
const SAMPLES = 4;
const BASE_Y = -0.25;      // bottom of the block (unexaggerated)
const WATER_Y = 0.07;      // reservoir level in terrain mode (unexaggerated)
const UNIFORM_FLOATS = 80;

// ---------------------------------------------------------------- math (column-major)

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0]);
}

function ortho(l, r, b, t, n, f) {
  return new Float32Array([
    2 / (r - l), 0, 0, 0,
    0, 2 / (t - b), 0, 0,
    0, 0, 1 / (n - f), 0,
    -(r + l) / (r - l), -(t + b) / (t - b), n / (n - f), 1,
  ]);
}

function invert(m) {
  const inv = new Float32Array(16);
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[2] * m[14] - m[4] * m[3] * m[13] - m[12] * m[2] * m[6] + m[12] * m[3] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  for (let i = 0; i < 16; i++) inv[i] /= det;
  return inv;
}

function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}

// ---------------------------------------------------------------- UI state

const $ = (id) => document.getElementById(id);
const state = {
  mode: params.get('mode') === 'geology' ? 'geology' : 'terrain',
  seed: 3,
  exag: 1.3,
  fold: 0.15,
  throw: 0.07,
  sliceX: 1,
  sliceZ: 1,
  sun: 135,
  contours: false,
  contacts: true,
  intrusion: true,
  rotate: true,
};

const CAMERAS = {
  terrain: { yaw: -0.75, pitch: 0.3, dist: 1.2, target: [0.05, 0.1, 0.05] },
  geology: { yaw: -0.75, pitch: 0.5, dist: 4.4, target: [0.15, 0.0, 0] },
};
const cam = structuredClone(CAMERAS[state.mode]);

function setMode(mode) {
  state.mode = mode;
  Object.assign(cam, structuredClone(CAMERAS[mode]));
  document.body.dataset.mode = mode;
  for (const b of document.querySelectorAll('[data-set-mode]')) b.setAttribute('aria-pressed', b.dataset.setMode === mode);
  state.contours = mode === 'geology';
  $('contours').checked = state.contours;
}

function bindUI(onSeed) {
  const sliders = [
    ['exag', 'exagVal', (v) => `${v.toFixed(1)}×`],
    ['fold', 'foldVal', (v) => v.toFixed(2)],
    ['throw', 'throwVal', (v) => v.toFixed(2)],
    ['sliceX', 'sliceXVal', (v) => `${Math.round(((v + 1) / 2) * 100)}%`],
    ['sliceZ', 'sliceZVal', (v) => `${Math.round(((v + 1) / 2) * 100)}%`],
    ['sun', 'sunVal', (v) => `${Math.round(v)}°`],
  ];
  for (const [key, label, fmt] of sliders) {
    const el = $(key);
    el.value = state[key];
    const update = () => { state[key] = parseFloat(el.value); $(label).textContent = fmt(state[key]); };
    el.addEventListener('input', update);
    update();
  }
  for (const key of ['contours', 'contacts', 'intrusion', 'rotate']) {
    const el = $(key);
    el.checked = state[key];
    el.addEventListener('change', () => { state[key] = el.checked; });
  }
  for (const b of document.querySelectorAll('[data-set-mode]')) b.addEventListener('click', () => setMode(b.dataset.setMode));
  setMode(state.mode);
  $('seed').addEventListener('click', () => {
    state.seed = Math.floor(Math.random() * 1000);
    onSeed();
  });

  const legend = $('legend');
  [...UNITS.keys()].reverse().forEach((i) => {
    const u = UNITS[i];
    const row = document.createElement('li');
    row.innerHTML = `<span class="sw" style="background:${u.color}"></span><span class="nm">${u.name}</span><span class="ag">${u.age}</span>`;
    legend.appendChild(row);
  });
}

function bindCamera(canvas) {
  let drag = null;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    state.rotate = false;
    $('rotate').checked = false;
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.pan) {
      const s = cam.dist * 0.0012;
      const right = [Math.cos(cam.yaw), 0, -Math.sin(cam.yaw)];
      const fwd = [Math.sin(cam.yaw), 0, Math.cos(cam.yaw)];
      cam.target[0] -= (right[0] * dx + fwd[0] * dy) * s;
      cam.target[2] -= (right[2] * dx + fwd[2] * dy) * s;
    } else {
      cam.yaw -= dx * 0.006;
      cam.pitch = Math.min(1.45, Math.max(0.04, cam.pitch + dy * 0.005));
    }
  });
  const end = () => { drag = null; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    cam.dist = Math.min(8, Math.max(0.4, cam.dist * Math.exp(e.deltaY * 0.001)));
  }, { passive: false });
}

// ---------------------------------------------------------------- WebGPU

async function init() {
  const canvas = $('gpu');
  let generate = async () => {};
  bindUI(() => generate());
  bindCamera(canvas);
  if (!navigator.gpu) return fail('This browser does not support WebGPU.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  device.lost.then((info) => fail(`WebGPU device lost: ${info.message}`));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));

  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });

  // --- heightmap generation (compute)
  const heightBuf = device.createBuffer({
    size: GRID * GRID * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const genParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const genPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: HEIGHT_WGSL }), entryPoint: 'main' },
  });
  const genBG = device.createBindGroup({
    layout: genPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: heightBuf } },
      { binding: 1, resource: { buffer: genParams } },
    ],
  });

  let generating = false;
  generate = async () => {
    if (generating) return;
    generating = true;
    device.queue.writeBuffer(genParams, 0, new Float32Array([GRID, state.seed, 0, 0]));
    const staging = device.createBuffer({ size: GRID * GRID * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(genPipeline);
    pass.setBindGroup(0, genBG);
    pass.dispatchWorkgroups(Math.ceil(GRID / 8), Math.ceil(GRID / 8));
    pass.end();
    enc.copyBufferToBuffer(heightBuf, 0, staging, 0, GRID * GRID * 4);
    device.queue.submit([enc.finish()]);

    // normalise the raw noise into a predictable elevation range
    await staging.mapAsync(GPUMapMode.READ);
    const h = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    let lo = Infinity, hi = -Infinity;
    for (const v of h) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (let i = 0; i < h.length; i++) {
      const t = (h[i] - lo) / (hi - lo);
      h[i] = 0.03 + Math.pow(t, 1.4) * 0.36;
    }
    device.queue.writeBuffer(heightBuf, 0, h);
    generating = false;
  };

  // --- render resources
  const uniformBuf = device.createBuffer({ size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const shadowTex = device.createTexture({
    size: [SHADOW_SIZE, SHADOW_SIZE],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });

  const bgl0 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });
  const bgl1 = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ],
  });
  const bg0 = device.createBindGroup({
    layout: bgl0,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: heightBuf } },
    ],
  });
  const bg1 = device.createBindGroup({
    layout: bgl1,
    entries: [
      { binding: 0, resource: shadowTex.createView() },
      { binding: 1, resource: shadowSampler },
    ],
  });

  const module = device.createShaderModule({ code: RENDER_WGSL });
  const info = await module.getCompilationInfo();
  for (const m of info.messages) console[m.type === 'error' ? 'error' : 'warn'](`WGSL ${m.lineNum}:${m.linePos} ${m.message}`);

  const shadowLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0] });
  const mainLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] });

  const shadowPipe = (vs) => device.createRenderPipeline({
    layout: shadowLayout,
    vertex: { module, entryPoint: vs },
    fragment: { module, entryPoint: 'fsShadow', targets: [] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  });
  const mainPipe = (vs, fs, blend, depthCompare = 'less') => device.createRenderPipeline({
    layout: mainLayout,
    vertex: { module, entryPoint: vs },
    fragment: {
      module,
      entryPoint: fs,
      targets: [{
        format,
        blend: blend ? {
          color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
        } : undefined,
      }],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: !blend && depthCompare !== 'always', depthCompare },
    multisample: { count: SAMPLES },
  });

  const pipes = {
    shadowTerrain: shadowPipe('vsTerrainShadow'),
    shadowWalls: shadowPipe('vsWallsShadow'),
    terrain: mainPipe('vsTerrain', 'fsTerrain', false),
    walls: mainPipe('vsWalls', 'fsWalls', false),
    ground: mainPipe('vsGround', 'fsGround', true),
    sky: mainPipe('vsSky', 'fsSky', false, 'always'),
    water: mainPipe('vsWater', 'fsWater', false),
  };

  const terrainVerts = (GRID - 1) * (GRID - 1) * 6;
  const wallVerts = 4 * WALL_SEGMENTS * 6;

  let msaaTex, depthTex;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width === w && canvas.height === h && msaaTex) return;
    canvas.width = w;
    canvas.height = h;
    msaaTex?.destroy();
    depthTex?.destroy();
    msaaTex = device.createTexture({ size: [w, h], format, sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    depthTex = device.createTexture({ size: [w, h], format: 'depth24plus', sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  await generate();
  $('loading').remove();

  // Debug hook: render one frame offscreen and return it as a PNG data URL.
  // Useful where the canvas can't be read back (e.g. headless SwiftShader).
  let capture = null;
  window.__captureFrame = async () => {
    const w = canvas.width, h = canvas.height;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const tex = device.createTexture({ size: [w, h], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const buf = device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    await new Promise((resolve) => { capture = { tex, buf, bytesPerRow, resolve }; });
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const img = new ImageData(w, h);
    const bgra = format === 'bgra8unorm';
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * bytesPerRow + x * 4, o = (y * w + x) * 4;
        img.data[o] = src[i + (bgra ? 2 : 0)];
        img.data[o + 1] = src[i + 1];
        img.data[o + 2] = src[i + (bgra ? 0 : 2)];
        img.data[o + 3] = src[i + 3];
      }
    buf.unmap();
    buf.destroy();
    tex.destroy();
    const out = new OffscreenCanvas(w, h);
    out.getContext('2d').putImageData(img, 0, 0);
    const blob = await out.convertToBlob({ type: 'image/png' });
    return new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  };

  const u = new Float32Array(UNIFORM_FLOATS);
  let last = performance.now();

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    resize();
    if (state.rotate) cam.yaw += dt * (state.mode === 'terrain' ? 0.04 : 0.12);

    const eye = [
      cam.target[0] + cam.dist * Math.cos(cam.pitch) * Math.sin(cam.yaw),
      cam.target[1] + cam.dist * Math.sin(cam.pitch),
      cam.target[2] + cam.dist * Math.cos(cam.pitch) * Math.cos(cam.yaw),
    ];
    const view = lookAt(eye, cam.target, [0, 1, 0]);
    const proj = perspective(0.6, canvas.width / canvas.height, 0.05, 30);

    const az = (state.sun * Math.PI) / 180;
    const terrain = state.mode === 'terrain';
    const el = terrain ? 0.5 : 0.75;
    const lightDir = [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
    const lightView = lookAt(lightDir.map((v) => v * 4), [0, 0, 0], [0, 1, 0]);
    const lightProj = ortho(-1.9, 1.9, -1.9, 1.9, 0.5, 8);

    const viewProj = mul(proj, view);
    u.set(viewProj, 0);
    u.set(mul(lightProj, lightView), 16);
    u.set([...eye, 1], 32);
    u.set([...lightDir, 0], 36);
    u.set(terrain ? [-1, 1, -1, 1] : [-1, state.sliceX, -1, state.sliceZ], 40);
    u.set([state.exag, BASE_Y, state.contours ? 1 : 0, GRID], 44);
    u.set([state.fold, state.throw, state.intrusion ? 1 : 0, state.seed], 48);
    u.set([WALL_SEGMENTS, state.contacts ? 1 : 0, now / 1000, terrain ? 1 : 0], 52);
    u.set(invert(viewProj), 56);
    u.set([WATER_Y, 0, 0, 0], 72);
    device.queue.writeBuffer(uniformBuf, 0, u);

    const enc = device.createCommandEncoder();

    const sp = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: { view: shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    sp.setBindGroup(0, bg0);
    sp.setPipeline(pipes.shadowTerrain);
    sp.draw(terrainVerts);
    if (!terrain) {
      sp.setPipeline(pipes.shadowWalls);
      sp.draw(wallVerts);
    }
    sp.end();

    const mp = enc.beginRenderPass({
      colorAttachments: [{
        view: msaaTex.createView(),
        resolveTarget: (capture?.tex ?? ctx.getCurrentTexture()).createView(),
        clearValue: [0, 0, 0, 0],
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    mp.setBindGroup(0, bg0);
    mp.setBindGroup(1, bg1);
    if (terrain) {
      mp.setPipeline(pipes.sky);
      mp.draw(3);
    }
    mp.setPipeline(pipes.terrain);
    mp.draw(terrainVerts);
    if (terrain) {
      mp.setPipeline(pipes.water);
      mp.draw(6);
    } else {
      mp.setPipeline(pipes.walls);
      mp.draw(wallVerts);
      mp.setPipeline(pipes.ground);
      mp.draw(6);
    }
    mp.end();

    if (capture) {
      const { tex, buf, bytesPerRow, resolve } = capture;
      enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [tex.width, tex.height]);
      device.queue.submit([enc.finish()]);
      capture = null;
      resolve();
    } else {
      device.queue.submit([enc.finish()]);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function fail(msg) {
  const el = $('loading');
  if (el) {
    el.textContent = msg;
    el.classList.add('error');
  }
  console.error(msg);
}

init().catch((e) => fail(e.message));
