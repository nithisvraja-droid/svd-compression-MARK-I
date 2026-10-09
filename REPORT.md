# Image Compression with the Singular Value Decomposition

**Course:** Mathematics for Intelligent Systems – I, School of AI, Amrita Vishwa Vidyapeetham, Coimbatore (2026)
**Team:** Nithis V. R. · Pragna · Dimple · Pavitraa

## 1. Aim

Store a photo with far fewer numbers while keeping it recognisable, using the singular value decomposition (SVD), and measure honestly how good that is: against theory, against JPEG, and against human and machine perception.

## 2. Theory

### 2.1 SVD of an image

A colour image of *m × n* pixels is three real matrices (R, G, B), each *m × n* with entries in [0, 1]. Every real matrix has a singular value decomposition

```
A = U Σ Vᵀ = Σᵢ σᵢ uᵢ vᵢᵀ,     σ₁ ≥ σ₂ ≥ … ≥ σᵣ > 0,  r = rank(A)
```

with orthonormal columns uᵢ ∈ ℝᵐ and vᵢ ∈ ℝⁿ. Each term σᵢuᵢvᵢᵀ is a rank-one "layer".

### 2.2 Best low-rank approximation

The truncation A_k = Σ_{i≤k} σᵢuᵢvᵢᵀ is the closest rank-k matrix to A (Eckart–Young–Mirsky):

```
‖A − A_k‖_F = √(σ²ₖ₊₁ + … + σ²ᵣ)        ‖A − A_k‖₂ = σₖ₊₁
```

### 2.3 Storage

A_k needs k columns of U (km numbers), k rows of Vᵀ (kn) and k singular values: k(m + n + 1) numbers against mn for the raw channel. The SVD form is smaller only while

```
k < mn / (m + n + 1)        (break-even rank)
```

e.g. k < 255 for 512 × 512. If the factors are stored as 4-byte floats while pixels are 1-byte integers, the break-even rank drops four-fold.

### 2.4 Energy

The fraction of the "energy" (squared Frobenius norm) kept by k layers is (σ₁² + … + σ_k²) / Σσᵢ². It is often used to choose k.

## 3. Method

| Step | Detail |
|---|---|
| Data | Seven scikit-image test images: astronaut, coffee, chelsea, rocket, text, gravel, colorwheel |
| SVD | `numpy.linalg.svd` per channel; randomized SVD (Halko et al.) for speed tests |
| Quality | relative Frobenius error, PSNR (dB), SSIM |
| Colour | per-channel RGB vs. stacked [R G B] vs. YCbCr with 60% of the layer budget on brightness, all at equal storage |
| File format | `.svdz`: U, Vᵀ quantised to int8 with one float32 scale per vector, σ as float32, zlib |
| Baselines | block SVD on b×b tiles; JPEG (Pillow) at the highest quality that fits the same byte budget |
| Denoising | Gaussian noise σ = 0.1; truncation at every k and at the Gavish–Donoho optimal threshold |
| Machine perception | SCRFD-500M face detector and ArcFace MobileFaceNet (InsightFace `buffalo_sc`) on every rank |
| Human perception | web test: ascending method of limits (k = 1…50) and 24 two-alternative forced-choice trials (k = 10…200) |

All code is in `svdlab/` and `scripts/`; `python scripts/run_experiments.py` regenerates every number below.

## 4. Results

**4.1 Theory checks out.** For the astronaut red channel the measured errors match the Eckart–Young predictions to within 3 × 10⁻¹⁴ for every k tested (`results/02_eckart_young.csv`).

**4.2 Spectra decide everything.** Singular values drop by 100–1000× within a few dozen layers, at very different speeds per image. The flat colour wheel needs 2 layers for 99% energy; gravel needs 77. The text image is not even full rank (162 of 172), because some rows of pixels repeat exactly.

**4.3 Energy is misleading.** Layer 1 is mostly the average colour, so a blurry blob already keeps 80–97% of the energy. PSNR and SSIM separate the images far better.

**4.4 Quality vs storage.** For the astronaut, 7.8% of the numbers (k = 20) gives 22.2 dB PSNR and 19.6% (k = 50) gives 27.4 dB. Smooth and synthetic images compress best; texture worst.

**4.5 Colour.** At equal storage, YCbCr beat per-channel RGB by 0.6–2.6 dB on average, and stacking [R G B] sat in between. Brightness carries most of the detail people see.

**4.6 Bytes vs numbers.** In float64 the factors at k = 50 (1.23 MB) are larger than the raw astronaut image (0.79 MB). 8-bit quantisation brings this to 144 KB at a cost below 0.16 dB in all cases tested.

**4.7 Block SVD.** At equal storage a single global SVD matched or beat 8×8, 16×16 and 32×32 tiles, because per-tile vectors are expensive.

**4.8 JPEG.** At matched file sizes JPEG scored 3–20 dB (median 8.8 dB) higher PSNR on every image. JPEG's fixed DCT basis costs nothing to store and its quantisation is perceptually tuned; SVD must store a new basis for every picture.

**4.9 Denoising.** Truncation removes noise because noise spreads over all singular values. The Gavish–Donoho rank (41–43) raised PSNR from 20.7 dB to 23.8 dB, close to the best achievable 24.1 dB (k = 57).

**4.10 Randomized SVD.** The top 10 layers of an 872 × 1000 matrix came more than 20× faster than a full SVD, with 0.1% extra error.

**4.11 Machine perception (astronaut).** The detector finds a face (confidence ≥ 0.5) from k = 15, ArcFace matches the identity (cosine ≥ 0.40) from k = 19, and SSIM reaches 0.95 only at k = 200. Recognition is therefore possible with about 7% of the numbers, while "indistinguishable from the original" needs about 78%. On the portrait used in our slides the thresholds were 7, 13 and 120. The detector is not monotonic in k: it briefly fires weakly at k = 5–6, then drops before locking on.

**4.12 Human perception.** The test is live at `docs/benchmark/`; results will be added when data collection is complete.

## 5. Discussion

SVD compression is the cleanest illustration of low-rank approximation: the theory predicts the error exactly, and the singular value spectrum explains which images compress well. As a practical codec it loses clearly to JPEG, and its storage advantage depends on how the factors are stored. Where SVD shines is understanding: the layers, the residual and the spectrum make the idea of "keeping what matters" concrete, and the same truncation removes noise.

The perception benchmark shows a large gap between "recognisable" and "identical": a face can be detected and identified at ranks that a pixel metric like SSIM still rates as poor. Whether humans follow the machine curves is the open question the human test answers.

## 6. Limitations and future work

- One face image in the benchmark; self-reported recognition. A lineup with foil faces would make Part 1 objective.
- Display size strongly affects the forced-choice task.
- Entropy coding of the quantised factors and adaptive per-channel ranks could narrow the JPEG gap.
- Eigenfaces (SVD across many face images) would connect this project directly to face recognition.

## References

See README.md.
