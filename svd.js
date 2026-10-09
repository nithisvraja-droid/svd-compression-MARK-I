/* Thin SVD in plain JavaScript (one-sided Jacobi / Hestenes method).
 *
 * Repeatedly rotates pairs of columns until every pair is orthogonal. The
 * column norms are then the singular values. Simple, accurate, and fast
 * enough for images up to ~512 px in a Web Worker.
 *
 * Works in three places: a Web Worker (postMessage API), a normal page
 * (window.SVD) and Node (module.exports, used by the tests).
 */
(function (root) {
  "use strict";

  // A: Float64Array, row-major, m x n. Returns {m, n, r, s, U, V}
  //   s: Float64Array(r) singular values, descending
  //   U: Float64Array(r*m)  - U[i*m + row]  = i-th left singular vector
  //   V: Float64Array(r*n)  - V[i*n + col]  = i-th right singular vector
  function svd(A, m, n, opts) {
    opts = opts || {};
    const tol = opts.tol || 1e-10, maxSweeps = opts.maxSweeps || 40, onSweep = opts.onSweep;
    const tr = m < n;                         // work on the tall orientation
    const M = tr ? n : m, N = tr ? m : n;
    // W: N columns of length M (column-major), V: N x N identity
    const W = new Float64Array(M * N), V = new Float64Array(N * N);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      const v = A[i * n + j];
      if (tr) W[i * M + j] = v; else W[j * M + i] = v;
    }
    for (let i = 0; i < N; i++) V[i * N + i] = 1;

    for (let sweep = 0; sweep < maxSweeps; sweep++) {
      let rotations = 0;
      for (let p = 0; p < N - 1; p++) {
        const po = p * M, pv = p * N;
        for (let q = p + 1; q < N; q++) {
          const qo = q * M;
          let a = 0, b = 0, g = 0;
          for (let i = 0; i < M; i++) {
            const x = W[po + i], y = W[qo + i];
            a += x * x; b += y * y; g += x * y;
          }
          if (Math.abs(g) <= tol * Math.sqrt(a * b) || g === 0) continue;
          rotations++;
          const zeta = (b - a) / (2 * g);
          const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
          const c = 1 / Math.sqrt(1 + t * t), s = c * t;
          for (let i = 0; i < M; i++) {
            const x = W[po + i], y = W[qo + i];
            W[po + i] = c * x - s * y; W[qo + i] = s * x + c * y;
          }
          const qv = q * N;
          for (let i = 0; i < N; i++) {
            const x = V[pv + i], y = V[qv + i];
            V[pv + i] = c * x - s * y; V[qv + i] = s * x + c * y;
          }
        }
      }
      if (onSweep) onSweep(sweep + 1, rotations);
      if (rotations === 0) break;
    }

    // singular values = column norms; sort descending
    const norms = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let i = 0; i < M; i++) s += W[j * M + i] ** 2;
      norms[j] = Math.sqrt(s);
    }
    const order = Array.from({ length: N }, (_, i) => i).sort((x, y) => norms[y] - norms[x]);
    const r = N, S = new Float64Array(r);
    const Ul = new Float64Array(r * M), Vr = new Float64Array(r * N);
    order.forEach((j, k) => {
      const sv = norms[j];
      S[k] = sv;
      if (sv > 1e-12) for (let i = 0; i < M; i++) Ul[k * M + i] = W[j * M + i] / sv;
      for (let i = 0; i < N; i++) Vr[k * N + i] = V[j * N + i];
    });
    // undo the transpose: A^T = Ul S Vr^T  =>  A = Vr S Ul^T
    return tr ? { m, n, r, s: S, U: Vr, V: Ul } : { m, n, r, s: S, U: Ul, V: Vr };
  }

  // Add layers [from, to) of d into out (Float32Array/Float64Array m*n, row-major)
  function addLayers(d, out, from, to, sign) {
    const { m, n, s, U, V } = d;
    sign = sign === undefined ? 1 : sign;
    for (let k = from; k < Math.min(to, d.r); k++) {
      const sk = sign * s[k], uo = k * m, vo = k * n;
      for (let i = 0; i < m; i++) {
        const ui = sk * U[uo + i];
        if (ui === 0) continue;
        const ro = i * n;
        for (let j = 0; j < n; j++) out[ro + j] += ui * V[vo + j];
      }
    }
    return out;
  }

  const api = { svd, addLayers };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SVD = api;

  // Web Worker entry: {id, channels: [Float64Array...], m, n}
  if (typeof window === "undefined" && typeof self !== "undefined" && typeof importScripts === "function") {
    self.onmessage = (e) => {
      const { id, channels, m, n } = e.data;
      const out = [];
      channels.forEach((A, c) => {
        out.push(svd(A, m, n, {
          tol: 1e-9,
          onSweep: (sw, rot) => self.postMessage({ id, progress: { channel: c, sweep: sw, rotations: rot } }),
        }));
      });
      self.postMessage({ id, result: out });
    };
  }
})(typeof self !== "undefined" ? self : this);
