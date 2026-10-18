import { UNITS, hexToRgb } from './geology.js';

const NOISE = /* wgsl */ `
fn hash2i(p: vec2i) -> f32 {
  var h = u32(p.x) * 374761393u + u32(p.y) * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}

fn vnoise(p: vec2f) -> f32 {
  let i = vec2i(floor(p));
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  let a = hash2i(i);
  let b = hash2i(i + vec2i(1, 0));
  let c = hash2i(i + vec2i(0, 1));
  let d = hash2i(i + vec2i(1, 1));
  return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

fn fbm(p0: vec2f, oct: i32) -> f32 {
  var p = p0;
  var sum = 0.0;
  var amp = 0.5;
  for (var i = 0; i < oct; i++) {
    sum += amp * vnoise(p);
    p = mat2x2f(1.6, 1.2, -1.2, 1.6) * p + vec2f(3.1, 7.7);
    amp *= 0.5;
  }
  return sum;
}

fn ridged(p0: vec2f, oct: i32) -> f32 {
  var p = p0;
  var sum = 0.0;
  var amp = 0.5;
  var prev = 1.0;
  for (var i = 0; i < oct; i++) {
    var r = 1.0 - abs(vnoise(p) * 2.0 - 1.0);
    r = r * r;
    sum += amp * r * prev;
    prev = r;
    p = mat2x2f(1.6, 1.2, -1.2, 1.6) * p + vec2f(5.3, 1.9);
    amp *= 0.5;
  }
  return sum;
}
`;

export const HEIGHT_WGSL = /* wgsl */ `
${NOISE}
@group(0) @binding(0) var<storage, read_write> outH: array<f32>;
@group(0) @binding(1) var<uniform> cp: vec4f; // N, seed

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let n = u32(cp.x);
  if (id.x >= n || id.y >= n) { return; }
  let p = (vec2f(id.xy) / f32(n - 1)) * 2.0 - 1.0;
  let s = vec2f(cp.y * 17.13, cp.y * 9.71);

  // domain-warped fbm for the broad landforms
  let q = vec2f(fbm(p * 1.3 + s, 5), fbm(p * 1.3 + s + vec2f(5.2, 1.3), 5));
  let base = fbm(p * 1.1 + q * 0.9 + s, 7);
  // ridged noise gives sharp mountain crests on the highs
  let r = ridged(p * 1.8 + q * 0.7 + s + vec2f(2.0, 4.0), 7);
  var h = base * 0.8 + r * 0.35 * smoothstep(0.4, 0.7, base);

  // a meandering river valley cuts across the block
  let riverZ = 0.25 * sin(p.x * 2.3 + cp.y) + 0.12 * (fbm(vec2f(p.x * 2.0, cp.y), 4) - 0.5);
  let dv = abs(p.y - riverZ);
  h -= 0.22 * exp(-dv * dv * 40.0);

  outH[id.y * n + id.x] = h;
}
`;

const palette = UNITS.map((u) => {
  const [r, g, b] = hexToRgb(u.color);
  return `vec3f(${r.toFixed(4)}, ${g.toFixed(4)}, ${b.toFixed(4)})`;
}).join(',\n    ');

export const RENDER_WGSL = /* wgsl */ `
${NOISE}

struct U {
  viewProj: mat4x4f,
  lightViewProj: mat4x4f,
  camPos: vec4f,
  lightDir: vec4f,
  clip: vec4f,   // xmin, xmax, zmin, zmax
  p0: vec4f,     // exaggeration, baseY, contours, N
  geo: vec4f,    // foldAmp, faultThrow, intrusion, seed
  p1: vec4f,     // wallSegments, contacts, time, -
};

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;

fn gridN() -> i32 { return i32(u.p0.w); }

fn H(i: i32, j: i32) -> f32 {
  let n = gridN();
  return heights[clamp(j, 0, n - 1) * n + clamp(i, 0, n - 1)];
}

// Height on the triangulated surface (matches the mesh exactly, so the walls seal it).
fn heightTri(x: f32, z: f32) -> f32 {
  let m = f32(gridN() - 1);
  let gx = clamp((x * 0.5 + 0.5) * m, 0.0, m - 0.0001);
  let gz = clamp((z * 0.5 + 0.5) * m, 0.0, m - 0.0001);
  let i = i32(floor(gx));
  let j = i32(floor(gz));
  let fx = gx - f32(i);
  let fz = gz - f32(j);
  if (fx + fz <= 1.0) {
    let h00 = H(i, j);
    return h00 + fx * (H(i + 1, j) - h00) + fz * (H(i, j + 1) - h00);
  }
  let h11 = H(i + 1, j + 1);
  return h11 + (1.0 - fx) * (H(i, j + 1) - h11) + (1.0 - fz) * (H(i + 1, j) - h11);
}

struct TV { p: vec3f, n: vec3f };

fn terrainVertex(vi: u32) -> TV {
  let n = gridN();
  let cells = u32(n - 1);
  let cell = vi / 6u;
  let k = vi % 6u;
  let ox = select(0, 1, k == 1u || k == 3u || k == 4u);
  let oz = select(0, 1, k == 2u || k == 4u || k == 5u);
  let i = i32(cell % cells) + ox;
  let j = i32(cell / cells) + oz;
  let step = 2.0 / f32(n - 1);
  let ex = u.p0.x;
  let dhdx = (H(i + 1, j) - H(i - 1, j)) / (2.0 * step) * ex;
  let dhdz = (H(i, j + 1) - H(i, j - 1)) / (2.0 * step) * ex;
  let p = vec3f(f32(i) * step - 1.0, H(i, j) * ex, f32(j) * step - 1.0);
  return TV(p, normalize(vec3f(-dhdx, 1.0, -dhdz)));
}

fn wallVertex(vi: u32) -> TV {
  let m = u32(u.p1.x);
  let per = m * 6u;
  let w = vi / per;
  let r = vi % per;
  let seg = r / 6u;
  let k = r % 6u;
  let ox = select(0.0, 1.0, k == 1u || k == 3u || k == 4u);
  let top = k == 2u || k == 4u || k == 5u;
  let t = (f32(seg) + ox) / f32(m);
  let c = u.clip;
  var x = c.x;
  var z = c.z;
  var nrm = vec3f(0.0, 0.0, -1.0);
  if (w == 0u) {
    x = mix(c.x, c.y, t);
  } else if (w == 1u) {
    x = mix(c.x, c.y, t); z = c.w; nrm = vec3f(0.0, 0.0, 1.0);
  } else if (w == 2u) {
    z = mix(c.z, c.w, t); nrm = vec3f(-1.0, 0.0, 0.0);
  } else {
    x = c.y; z = mix(c.z, c.w, t); nrm = vec3f(1.0, 0.0, 0.0);
  }
  let y = select(u.p0.y, heightTri(x, z), top) * u.p0.x;
  return TV(vec3f(x, y, z), nrm);
}

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) wp: vec3f,
  @location(1) nrm: vec3f,
};

@vertex fn vsTerrain(@builtin(vertex_index) vi: u32) -> VOut {
  let t = terrainVertex(vi);
  return VOut(u.viewProj * vec4f(t.p, 1.0), t.p, t.n);
}

@vertex fn vsWalls(@builtin(vertex_index) vi: u32) -> VOut {
  let t = wallVertex(vi);
  return VOut(u.viewProj * vec4f(t.p, 1.0), t.p, t.n);
}

@vertex fn vsTerrainShadow(@builtin(vertex_index) vi: u32) -> VOut {
  let t = terrainVertex(vi);
  return VOut(u.lightViewProj * vec4f(t.p, 1.0), t.p, t.n);
}

@vertex fn vsWallsShadow(@builtin(vertex_index) vi: u32) -> VOut {
  let t = wallVertex(vi);
  return VOut(u.lightViewProj * vec4f(t.p, 1.0), t.p, t.n);
}

fn outsideClip(p: vec3f) -> bool {
  let c = u.clip;
  return p.x < c.x || p.x > c.y || p.z < c.z || p.z > c.w;
}

@fragment fn fsShadow(in: VOut) {
  if (outsideClip(in.wp)) { discard; }
}

// ---------------------------------------------------------------- geology

fn palette(unit: i32) -> vec3f {
  var pal = array<vec3f, ${UNITS.length}>(
    ${palette}
  );
  return pal[clamp(unit, 0, ${UNITS.length - 1})];
}

struct Geo { unit: i32, contact: f32, fault: f32 };

// p: x, z in [-1, 1]; y unexaggerated elevation
fn geology(p: vec3f) -> Geo {
  let so = vec2f(u.geo.w * 13.7, u.geo.w * 7.1);
  let xz = p.xz;

  // erosional unconformity: flat-lying cover rests on the folded sequence
  let unc = 0.33 - 0.03 * xz.x + 0.035 * (vnoise(xz * 3.0 + so) - 0.5);

  // dipping normal fault
  let fdir = normalize(vec2f(0.9, -0.45));
  let fd = dot(xz, fdir) - 0.1 + p.y * 0.35;

  // plunging folds, rotated to strike NE
  let ca = cos(0.55);
  let sa = sin(0.55);
  let uu = ca * xz.x + sa * xz.y;
  let vv = -sa * xz.x + ca * xz.y;
  let fold = u.geo.x * sin(uu * 4.4 + 0.9 * sin(vv * 2.3) + 0.5) * (0.75 + 0.35 * vv)
           + 0.03 * uu + 0.05 * (vnoise(xz * 2.5 + so) - 0.5);
  var s = p.y - fold;
  if (fd > 0.0) { s += u.geo.y; }

  var bounds = array<f32, 8>(-0.15, -0.09, -0.035, 0.02, 0.08, 0.14, 0.21, 0.29);
  var unit = 0;
  var contact = 1e9;
  for (var k = 0; k < 8; k++) {
    if (s > bounds[k]) { unit = k + 1; }
    contact = min(contact, abs(s - bounds[k]));
  }
  var fault = abs(fd) * 0.9;

  // granite pluton (post-dates the fault)
  if (u.geo.z > 0.5) {
    let q = (p - vec3f(0.38, -0.06, -0.32)) / vec3f(0.32, 0.36, 0.26);
    let isdf = (length(q) - 1.0) * 0.26 + 0.035 * (vnoise(xz * 6.0 + so + 3.0) - 0.5);
    if (isdf < 0.0) {
      unit = 11; contact = -isdf; fault = 1.0;
    } else {
      contact = min(contact, isdf);
    }
  }

  // dolerite dyke (youngest intrusion)
  let dn = normalize(vec2f(0.35, 1.0));
  let dd = abs(dot(xz - vec2f(-0.35, 0.1), dn) + 0.04 * p.y) - 0.012;
  if (dd < 0.0) {
    unit = 12; contact = -dd; fault = 1.0;
  } else {
    contact = min(contact, dd);
  }

  // cover sequence above the unconformity
  if (p.y > unc) {
    let cb = unc + 0.06;
    unit = select(9, 10, p.y > cb);
    contact = min(p.y - unc, abs(p.y - cb));
    fault = 1.0;
  } else {
    contact = min(contact, unc - p.y);
  }
  return Geo(unit, contact, fault);
}

// ---------------------------------------------------------------- lighting

fn shadowAt(wp: vec3f, n: vec3f) -> f32 {
  let lp = u.lightViewProj * vec4f(wp + n * 0.006, 1.0);
  let ndc = lp.xyz / lp.w;
  let uv = ndc.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z > 1.0) { return 1.0; }
  let texel = 1.0 / f32(textureDimensions(shadowMap).x);
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum += textureSampleCompareLevel(shadowMap, shadowSamp, uv + vec2f(f32(x), f32(y)) * texel * 1.5, ndc.z - 0.0015);
    }
  }
  return sum / 9.0;
}

fn light(base: vec3f, n: vec3f, wp: vec3f) -> vec3f {
  let L = normalize(u.lightDir.xyz);
  let ndl = max(dot(n, L), 0.0);
  let sh = shadowAt(wp, n);
  let sky = 0.5 + 0.5 * n.y;
  let amb = mix(vec3f(0.36, 0.33, 0.36), vec3f(0.60, 0.64, 0.70), sky);
  let V = normalize(u.camPos.xyz - wp);
  let spec = pow(max(dot(n, normalize(L + V)), 0.0), 40.0) * 0.08 * sh;
  return base * (amb * 0.8 + vec3f(1.0, 0.95, 0.86) * ndl * sh * 0.72) + spec;
}

fn lineMask(d: f32, w0: f32, w1: f32) -> f32 {
  let fw = max(fwidth(d), 1e-6);
  return 1.0 - smoothstep(w0, w1, d / fw);
}

@fragment fn fsTerrain(in: VOut) -> @location(0) vec4f {
  let ex = u.p0.x;
  let gp = vec3f(in.wp.x, in.wp.y / ex, in.wp.z);
  let g = geology(gp);
  let n = normalize(in.nrm);

  var base = palette(g.unit) * (0.93 + 0.07 * vnoise(gp.xz * 70.0));
  var col = light(base, n, in.wp);

  let cl = lineMask(g.contact, 0.5, 1.3);
  let fl = lineMask(g.fault, 0.9, 2.0);
  let hc = gp.y / 0.025;
  let fh = max(fwidth(hc), 1e-6);
  let minor = 1.0 - smoothstep(0.3, 1.0, abs(fract(hc - 0.5) - 0.5) / fh);
  let major = 1.0 - smoothstep(0.6, 1.5, abs(fract(hc / 5.0 - 0.5) - 0.5) * 5.0 / fh);

  col = mix(col, col * 0.62, max(minor * 0.25, major * 0.55) * u.p0.z);
  col = mix(col, col * 0.3, cl * 0.85 * u.p1.y);
  col = mix(col, vec3f(0.55, 0.07, 0.09), fl * u.p1.y);

  if (outsideClip(in.wp)) { discard; }
  return vec4f(col, 1.0);
}

@fragment fn fsWalls(in: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4f {
  let ex = u.p0.x;
  let gp = vec3f(in.wp.x, in.wp.y / ex, in.wp.z);
  let g = geology(gp);
  let n = select(-in.nrm, in.nrm, ff);

  // fine bedding texture
  let along = in.wp.x + in.wp.z;
  let bed = 0.9 + 0.1 * vnoise(vec2f(along * 18.0, gp.y * 260.0));
  var col = light(palette(g.unit) * bed * 0.94, n, in.wp);

  let cl = lineMask(g.contact, 0.5, 1.3);
  let fl = lineMask(g.fault, 0.9, 2.0);
  col = mix(col, col * 0.3, cl * 0.85 * u.p1.y);
  col = mix(col, vec3f(0.55, 0.07, 0.09), fl * u.p1.y);

  // soft darkening towards the base of the block
  let depth = clamp((in.wp.y - u.p0.y * ex) / (0.25 * ex), 0.0, 1.0);
  col *= 0.8 + 0.2 * depth;
  return vec4f(col, 1.0);
}

// ---------------------------------------------------------------- ground shadow

struct GOut { @builtin(position) pos: vec4f, @location(0) wp: vec3f };

@vertex fn vsGround(@builtin(vertex_index) vi: u32) -> GOut {
  var c = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, 1));
  let p = vec3f(c[vi].x * 3.0, u.p0.y * u.p0.x - 0.003, c[vi].y * 3.0);
  return GOut(u.viewProj * vec4f(p, 1.0), p);
}

@fragment fn fsGround(in: GOut) -> @location(0) vec4f {
  let c = u.clip;
  let center = vec2f(c.x + c.y, c.z + c.w) * 0.5;
  let half = vec2f(c.y - c.x, c.w - c.z) * 0.5;
  let q = abs(in.wp.xz - center) - half;
  let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
  let sh = shadowAt(in.wp, vec3f(0.0, 1.0, 0.0));
  let fade = 1.0 - smoothstep(0.4, 2.6, length(in.wp.xz));
  let a = clamp((0.22 * (1.0 - sh) + 0.38 * exp(-max(d, 0.0) * 18.0)) * fade, 0.0, 0.6);
  return vec4f(vec3f(0.12, 0.09, 0.07) * a, a);
}
`;
