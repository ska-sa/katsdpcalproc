"""Tests for :mod:`katsdpcalproc.calprocs_dask`."""

import unittest

import numpy as np
import dask.array as da

from katsdpcalproc import calprocs, calprocs_dask

import test_calprocs


def as_dask(arr):
    return da.from_array(arr, chunks=arr.shape)


def unit(value):
    return value / np.abs(value)


class TestStefcal(test_calprocs.TestStefcal):
    """Tests for :func:`katsdpcalproc.calprocs_dask.stefcal`."""
    def _call_stefcal(self, rawvis, num_ants, corrprod_lookup, weights=None,
                      ref_ant=0, init_gain=None, *args, **kwargs):
        rawvis = da.asarray(rawvis)
        if weights is not None:
            weights = da.asarray(weights)
        if init_gain is not None:
            init_gain = da.asarray(init_gain)
        return calprocs_dask.stefcal(rawvis, num_ants, corrprod_lookup, weights, ref_ant,
                                     init_gain, *args, **kwargs).compute()

    def _wrap_array(self, array):
        # Overloads the base class to convert arrays to dask arrays
        if array is None:
            return None
        elif array.ndim >= 1 and array.shape[0] == 8192:
            # It's a timing test. Do something sensible to get good performance
            return da.from_array(array, chunks=(512,) + array.shape[1:])
        else:
            # Give it some random chunking, to check that the chunks of
            # multiple arrays get mapped together correctly.
            chunks = []
            for s in array.shape:
                if s > 1:
                    chunks.append(self.random_state.randint(1, s - 1))
                else:
                    chunks.append(1)
            return da.from_array(array, chunks=tuple(chunks))


class TestAlignChunks(unittest.TestCase):
    def test_simple(self):
        old = ((15, 18, 15, 16, 8),)   # Boundaries: 0, 15, 33, 48, 64, 72
        self.assertEqual(((12, 4, 16, 4, 12, 16, 8),), calprocs_dask._align_chunks(old, {0: 4}))

    def test_already_aligned(self):
        old = ((5, 10, 15, 5),)
        self.assertEqual(old, calprocs_dask._align_chunks(old, {0: 5}))

    def test_small_chunks(self):
        old = ((1,) * 12,)
        self.assertEqual(((5, 5, 2),), calprocs_dask._align_chunks(old, {0: 5}))

    def test_align_1(self):
        old = ((1, 2, 3, 4, 1),)
        self.assertEqual(old, calprocs_dask._align_chunks(old, {0: 1}))

    def test_multi_dimensions(self):
        old = ((5, 5, 5), (4, 4, 4), (7, 7, 7))
        self.assertEqual(((4, 4, 4, 3), (4, 4, 4), (6, 2, 6, 6, 1)),
                         calprocs_dask._align_chunks(old, {0: 4, 2: 2}))


class TestWavg(unittest.TestCase):
    """Tests for :func:`katsdpcalproc.calprocs_dask.wavg`"""
    def setUp(self):
        shape = (10, 5, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)

        # Put in some NaNs and flags to check that they're handled correctly
        # Put in some zero visibilities and high weights
        self.data[:, 0, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 0j, 6, 7, 8, 9]
        self.weights[:, 0, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 1e19, 5, 1e20]
        self.flags[:, 0, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => Zeros in output
        self.data[:, 3, 2, 2] = np.nan
        self.flags[:, 4, 0, 3] = 4

        # A completely zero column and a column with very high weights => Zeros in output
        self.data[:, 1, 0, 0] = 0j
        self.weights[:, 1, 1, 0] = 1e19

        self.expected = np.ones((5, 3, 10), np.complex64)
        self.expected[0, 1, 1] = 4.5 + 0.5j
        self.expected[3, 2, 2] = 0j
        self.expected[4, 0, 3] = 0j
        self.expected[1, 0, 0] = 0j
        self.expected[1, 1, 0] = 0j

    def test_basic(self):
        # Check for all axes
        for i in [0, 1, 2, 3, -1, -2, -3, -4]:
            data_i = np.moveaxis(self.data, 0, i)
            flags_i = np.moveaxis(self.flags, 0, i)
            weights_i = np.moveaxis(self.weights, 0, i)

            actual = calprocs_dask.wavg(as_dask(data_i), as_dask(flags_i), as_dask(weights_i),
                                        axis=i)
            self.assertEqual(np.complex64, actual.dtype)
            np.testing.assert_allclose(self.expected, actual, rtol=1e-6)

    def test_full(self):
        expected_weights = 10 * np.ones((5, 3, 10), np.float32)
        expected_flags = np.zeros((5, 3, 10), np.bool_)

        expected_weights[0, 1, 1] = 4
        expected_weights[3, 2, 2] = 0
        expected_weights[4, 0, 3] = 0
        expected_weights[1, 0, 0] = 0
        expected_weights[1, 1, 0] = 0

        expected_flags[0, 1, 1] = True
        expected_flags[3, 2, 2] = True
        expected_flags[4, 0, 3] = True
        expected_flags[1, 0, 0] = True
        expected_flags[1, 1, 0] = True

        for i in [0, 1, 2, 3, -1, -2, -3, -4]:
            data_i = np.moveaxis(self.data, 0, i)
            flags_i = np.moveaxis(self.flags, 0, i)
            weights_i = np.moveaxis(self.weights, 0, i)

            actual_data, actual_flags, actual_weights = calprocs_dask.wavg_full(
                 as_dask(data_i), as_dask(flags_i), as_dask(weights_i), axis=i)
            np.testing.assert_allclose(expected_weights, actual_weights, rtol=1e-6)
            np.testing.assert_allclose(expected_flags, actual_flags)


class TestWavgFullT(unittest.TestCase):
    """Tests for :func:`katsdpcalproc.calprocs_dask.wavg_full_t`"""
    def setUp(self):
        shape = (10, 5, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)
        # Put in some NaNs, zeros, high weights and flags to check that they're handled correctly
        self.data[:, 0, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 5, 0, 7, 8, 9]
        self.weights[:, 0, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 2e19, 5, 6]
        self.flags[:, 0, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => Zeros in output
        self.data[:, 1, 2, 2] = np.nan
        self.flags[:, 2, 0, 3] = 4
        # A completely zero column and column with all high weights => Zeros in output
        self.data[:, 1, 2, 3] = 0j
        self.weights[:, 2, 0, 4] = 2e19

    def test_basic(self):
        out_shape = (3, 5, 3, 10)
        expected_data = np.ones(out_shape, np.complex64)
        expected_weights = np.ones(out_shape, np.float32) * 4
        expected_weights[2, ...] = 2    # Only two samples added together
        expected_flags = np.zeros(out_shape, np.bool_)
        expected_data[:, 0, 1, 1] = [2j, 5, 0j]
        expected_weights[:, 0, 1, 1] = [1, 2, 0]
        expected_flags[:, 0, 1, 1] = [False, False, True]

        expected_data[:, 1, 2, 2] = 0j
        expected_weights[:, 1, 2, 2] = 0
        expected_flags[:, 1, 2, 2] = True

        expected_data[:, 2, 0, 3] = 0j
        expected_weights[:, 2, 0, 3] = 0
        expected_flags[:, 2, 0, 3] = True

        expected_data[:, 1, 2, 3] = 0j
        expected_weights[:, 1, 2, 3] = 0
        expected_flags[:, 1, 2, 3] = True

        expected_data[:, 2, 0, 4] = 0j
        expected_weights[:, 2, 0, 4] = 0
        expected_flags[:, 2, 0, 4] = True

        out_data, out_flags, out_weights = calprocs_dask.wavg_full_t(
            self.data, self.flags, self.weights, 4)
        np.testing.assert_allclose(expected_data, out_data, rtol=1e-6)
        np.testing.assert_equal(expected_flags, out_flags)
        np.testing.assert_allclose(expected_weights, out_weights, rtol=1e-6)

    def test_threshold(self):
        """Test thresholding on flags"""
        # This assumes the threshold default is 0.3 - it's not currently
        # settable via wavg_full_t.
        self.flags[:2, 0, 0, 0] = 4
        self.flags[:4, 0, 0, 1] = 4
        out_data, out_flags, out_weights = calprocs_dask.wavg_full_t(
            self.data, self.flags, self.weights, 10, threshold=0.3)
        self.assertEqual(False, out_flags[0, 0, 0, 0])
        self.assertEqual(True, out_flags[0, 0, 0, 1])


class TestWavgFullF(unittest.TestCase):
    """Tests for :func:`katsdpcalproc.calprocs_dask.wavg_full_f`

    The tests just compare results against
    :func:`katsdpcalproc.calprocs.wavg_full_f` (i.e., the non-dask version),
    which is assumed to have its own tests.
    """
    def _test(self, shape, chunks, chanav):
        rs = np.random.RandomState(seed=1)
        data = rs.standard_normal(shape) + 1j * rs.standard_normal(shape)
        data = data.astype(np.complex64)
        weights = rs.uniform(size=shape).astype(np.float32)
        flags = rs.uniform(size=shape) < 0.05
        # Ensure some data values are NaN and some weights are zero
        data[rs.uniform(size=shape) < 0.1] = np.nan
        weights[rs.uniform(size=shape) < 0.1] = 0.0
        # Ensure some whole chunks are flagged/zero-weight/nan
        data[0, :, 0, 0] = np.nan
        weights[0, :, 0, 1] = 0.0
        flags[0, :, 0, 2] = True

        ex_data, ex_flags, ex_weights = calprocs.wavg_full_f(data, flags, weights, chanav)

        data = da.from_array(data, chunks=chunks)
        flags = da.from_array(flags, chunks=chunks)
        weights = da.from_array(weights, chunks=chunks)
        av_data, av_flags, av_weights = calprocs_dask.wavg_full_f(data, flags, weights, chanav)

        np.testing.assert_array_equal(ex_data, av_data.compute())
        np.testing.assert_array_equal(ex_flags, av_flags.compute())
        np.testing.assert_array_equal(ex_weights, av_weights.compute())

    def test_aligned(self):
        """Test where all chunks are aligned to chanav"""
        self._test((40, 32, 2, 6), (4, 8, 2, 6), 4)

    def test_overhang(self):
        """Test where the chunk boundaries are aligned, but there is a leftover piece"""
        self._test((40, 22, 2, 6), (4, (12, 4, 4, 2), 2, 3), 4)

    def test_unaligned(self):
        """Test where the chunk boundaries are not aligned at all"""
        self._test((40, 22, 2, 6), (4, (3, 6, 5, 8), 1, 6), 4)


class TestWavgAnt(unittest.TestCase):
    """Tests for :func:`katsdpcalproc.calprocs_dask.wavg_ant`"""
    def setUp(self):
        shape = (10, 5, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)
        self.bls_lookup = np.array([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
                                    (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

    def test_basic(self):
        # Put in some NaNs and flags to check that they're handled correctly
        # Put in some zero visibilities and high weights
        self.data[1, 0, 1, :] = [1 + 1j, 2j, np.nan, 4j, 0, 5, 6, 7j, 8, 9]
        self.weights[1, 0, 1, :] = [np.nan, 1, 2, 0, 3, 4, 2e15, 1, np.nan, 2]
        self.flags[1, 0, 1, :] = [4, 0, 0, 0, 0, 4, 0, 0, 0, 0]
        # A completely NaN column and a completely flagged column => Zeros in output
        self.data[3, 2, 2, :] = np.nan
        self.flags[4, 0, 2, :] = 4

        # A completely zero column and a column with very high weights => Zeros in output
        self.data[1, 0, 0, :] = 0j
        self.weights[1, 1, 0, :] = 1e19

        out_shape = (10, 5, 3, 5)
        expected_vis = np.ones(out_shape, np.complex64)
        expected_flags = np.zeros(out_shape, np.bool_)
        expected_weights = 4 * np.ones(out_shape, np.float32)

        expected_vis[1, 0, 1, :] = [2j, 0, 2.5j, (18-7j)/3, 9]
        expected_flags[1, 0, 1, :] = [False, True, False, False, False]
        expected_weights[1, 0, 1, :] = [1, 0, 2, 3, 2]

        expected_vis[3, 2, 2, :] = 0j
        expected_weights[3, 2, 2, :] = 0
        expected_flags[3, 2, 2, :] = True

        expected_vis[4, 0, 2, :] = 0j
        expected_weights[4, 0, 2, :] = 0
        expected_flags[4, 0, 2, :] = True

        expected_vis[1, 0, 0, :] = 0j
        expected_weights[1, 0, 0, :] = 0
        expected_flags[1, 0, 0, :] = True

        expected_vis[1, 1, 0, :] = 0j
        expected_weights[1, 1, 0, :] = 0
        expected_flags[1, 1, 0, :] = True

        out_vis, out_flags, out_weights = calprocs_dask.wavg_ant(
            as_dask(self.data), as_dask(self.flags), as_dask(self.weights),
            np.arange(5), self.bls_lookup)
        self.assertEqual(np.complex64, out_vis.dtype)
        np.testing.assert_allclose(expected_vis, out_vis, rtol=1e-6)
