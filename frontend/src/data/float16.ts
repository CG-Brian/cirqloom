// IEEE 754 half precision → JS number. Bundles store attention as little-endian float16
// (AGENTS.md §8 asset strategy); decoding by hand avoids depending on Float16Array support.

export function halfToNumber(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024); // zero and subnormals
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Decodes `count` little-endian float16 values starting at element `start`. */
export function decodeFloat16(buffer: ArrayBuffer, start: number, count: number): Float32Array {
  const view = new DataView(buffer);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = halfToNumber(view.getUint16((start + i) * 2, true));
  }
  return out;
}
