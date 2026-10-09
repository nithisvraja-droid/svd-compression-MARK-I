"""Tests. Run with ``python -m pytest`` (or ``python -m unittest``)."""
import unittest

import numpy as np

from svdlab import codec, colour
from svdlab.blocks import block_svd
from svdlab.core import (break_even_rank, compress, decompose, eckart_young_error, energy,
                         randomized_svd, rank_for_energy, reconstruct, storage_numbers, to_float,
                         to_uint8)
from svdlab.denoise import add_noise, denoise
from svdlab.jpeg import jpeg_at_size
from svdlab.metrics import psnr, rel_error, ssim

RNG = np.random.default_rng(42)


def smooth_image(m=48, n=64):
    """Low-rank-ish test picture: a few smooth gradients plus a little noise."""
    y, x = np.mgrid[0:1:m * 1j, 0:1:n * 1j]
    img = np.stack([0.5 + 0.4 * np.sin(3 * x) * y, 0.3 + 0.5 * x * y, 0.6 - 0.4 * y], 2)
    return np.clip(img + 0.01 * RNG.standard_normal(img.shape), 0, 1)


class TestCore(unittest.TestCase):
    def test_full_rank_is_exact(self):
        img = smooth_image()
        self.assertLess(np.abs(compress(img, 48) - img).max(), 1e-10)

    def test_error_decreases_with_k(self):
        img = smooth_image()
        errs = [rel_error(img, compress(img, k)) for k in (1, 2, 5, 10, 20)]
        self.assertTrue(all(a >= b for a, b in zip(errs, errs[1:])))

    def test_eckart_young(self):
        A = RNG.random((30, 40))
        d = decompose(A[:, :, None])[0]
        for k in (1, 5, 10, 25):
            R = A - d.reconstruct(k)
            fro, spec = eckart_young_error(d.s, k)
            self.assertAlmostEqual(np.linalg.norm(R), fro, places=10)
            self.assertAlmostEqual(np.linalg.norm(R, 2), spec, places=10)

    def test_energy(self):
        s = np.array([3.0, 2.0, 1.0])
        np.testing.assert_allclose(energy(s), [9 / 14, 13 / 14, 1.0])
        self.assertEqual(rank_for_energy(s, 0.9), 2)

    def test_storage(self):
        self.assertEqual(storage_numbers(388, 515, 20, 3), 3 * 20 * 904)
        self.assertEqual(break_even_rank(388, 515), 221)
        self.assertEqual(break_even_rank(388, 515, bytes_per_number=4), 55)

    def test_per_channel_k(self):
        img = smooth_image()
        rec = reconstruct(decompose(img), [1, 48, 48])
        self.assertLess(np.abs(rec[:, :, 1] - img[:, :, 1]).max(), 1e-10)

    def test_randomized_matches_exact(self):
        A = RNG.random((80, 10)) @ RNG.random((10, 120))  # exactly rank 10
        exact = decompose(A[:, :, None])[0]
        r = randomized_svd(A, 10)
        np.testing.assert_allclose(r.s, exact.s[:10], rtol=1e-8)
        np.testing.assert_allclose(r.reconstruct(10), A, atol=1e-8)

    def test_uint8_roundtrip(self):
        img = smooth_image()
        np.testing.assert_allclose(to_float(to_uint8(img)), img, atol=1 / 255)


class TestExtras(unittest.TestCase):
    def test_codec_roundtrip(self):
        img = smooth_image()
        data = codec.encode(img, 10)
        rec = codec.decode(data)
        self.assertEqual(rec.shape, img.shape)
        self.assertLess(abs(psnr(img, rec) - psnr(img, compress(img, 10))), 0.5)
        self.assertLess(len(data), img.size)  # smaller than raw 8-bit pixels

    def test_colour_modes_shape_and_quality(self):
        img = smooth_image()
        for fn in colour.METHODS.values():
            rec = fn(img, 5)
            self.assertEqual(rec.shape, img.shape)
            self.assertGreater(psnr(img, rec), 20)

    def test_ycbcr_roundtrip(self):
        img = smooth_image()
        np.testing.assert_allclose(colour.ycbcr_to_rgb(colour.rgb_to_ycbcr(img)), img, atol=1e-9)

    def test_block_full_rank_exact(self):
        img = smooth_image(50, 70)  # not a multiple of the block size
        np.testing.assert_allclose(block_svd(img, 16, 16), img, atol=1e-9)

    def test_jpeg_respects_budget(self):
        img = smooth_image(64, 64)
        _, _, size = jpeg_at_size(img, 4000)
        self.assertLessEqual(size, 4000)

    def test_denoise_helps(self):
        clean = smooth_image(96, 128)
        noisy = add_noise(clean, 0.1)
        rec, ks = denoise(noisy)
        self.assertGreater(psnr(clean, rec), psnr(clean, noisy))
        self.assertTrue(all(k >= 1 for k in ks))

    def test_ssim_identity(self):
        img = smooth_image()
        self.assertAlmostEqual(ssim(img, img), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
