// Checks the browser SVD (docs/svd.js): run with `node tests/svd_js.test.mjs`
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const { svd, addLayers } = require("../docs/svd.js");

function rand(m, n, seed = 1) {
  const A = new Float64Array(m * n);
  for (let i = 0; i < A.length; i++) { seed = (seed * 16807) % 2147483647; A[i] = seed / 2147483647; }
  return A;
}

for (const [m, n] of [[40, 30], [30, 50], [64, 64]]) {
  const A = rand(m, n);
  const d = svd(A, m, n);
  // full reconstruction is exact
  const R = addLayers(d, new Float64Array(m * n), 0, d.r);
  let err = 0;
  for (let i = 0; i < A.length; i++) err = Math.max(err, Math.abs(R[i] - A[i]));
  assert.ok(err < 1e-10, `reconstruction error ${err} for ${m}x${n}`);
  // singular values sorted, non-negative
  for (let i = 1; i < d.r; i++) assert.ok(d.s[i - 1] >= d.s[i] && d.s[i] >= 0);
  // Eckart-Young: Frobenius error of rank k = sqrt(sum of the tail)
  const k = 5, Rk = addLayers(d, new Float64Array(m * n), 0, k);
  let fro = 0; for (let i = 0; i < A.length; i++) fro += (A[i] - Rk[i]) ** 2;
  let tail = 0; for (let i = k; i < d.r; i++) tail += d.s[i] ** 2;
  assert.ok(Math.abs(Math.sqrt(fro) - Math.sqrt(tail)) < 1e-9);
  console.log(`ok ${m}x${n}`);
}
