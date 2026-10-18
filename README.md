# Geomap

A 3D geological block diagram rendered with raw WebGPU (no libraries).

- A compute shader generates the terrain heightmap (domain-warped fBm + ridged noise + a river valley).
- One geology function in WGSL is evaluated per fragment for both the top surface and the block sides, so
  the geological map and the cross-sections always agree: folded strata, a dipping normal fault,
  a granite pluton, a dolerite dyke, and flat-lying cover above an unconformity.
- Shadow-mapped sun, contours, contact lines, MSAA, and adjustable section slices.

## Run

WebGPU needs a secure context, so serve the folder over localhost:

```bash
python3 -m http.server 5173
```

Then open http://localhost:5173 in a WebGPU-capable browser (recent Chrome or Edge; Firefox and Safari behind flags on some platforms).

`?grid=256` (64 to 1024) changes the heightmap resolution; the default is 512.
