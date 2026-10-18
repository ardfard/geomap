// Stratigraphic column. Index = unit id used by the shader.
export const UNITS = [
  { name: 'Gneiss basement', age: 'Precambrian', color: '#c77d98' },
  { name: 'Quartzite', age: 'Cambrian', color: '#93b17a' },
  { name: 'Limestone', age: 'Ordovician', color: '#4fa38c' },
  { name: 'Graptolitic shale', age: 'Silurian', color: '#a9d3c0' },
  { name: 'Old Red Sandstone', age: 'Devonian', color: '#c98957' },
  { name: 'Reef limestone', age: 'L. Carboniferous', color: '#8ea4c8' },
  { name: 'Coal measures', age: 'U. Carboniferous', color: '#6f7b8e' },
  { name: 'Red marl', age: 'Permian', color: '#e48160' },
  { name: 'Mudstone', age: 'Triassic', color: '#b48ccb' },
  { name: 'Chalk', age: 'Cretaceous', color: '#c6da72' },
  { name: 'Clay & sand', age: 'Paleogene', color: '#f3d36f' },
  { name: 'Granite pluton', age: 'Intrusive', color: '#e5577c' },
  { name: 'Dolerite dyke', age: 'Intrusive', color: '#4d3542' },
];

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
