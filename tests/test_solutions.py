"""Tests for :mod:`katsdpcalproc.solutions`"""

import unittest

import numpy as np

from katsdpcalproc.solutions import (
    CalSolution, CalSolutionStore, CalSolutionStoreLatest
)


class TestCalSolutionStoreLatest(unittest.TestCase):
    make_store = CalSolutionStoreLatest

    def setUp(self):
        self.sol1 = CalSolution('G', np.arange(10), 1234.0, 'sol1')
        self.sol2 = CalSolution('G', np.arange(10, 20), 2345.0, 'sol2')
        self.sol3 = CalSolution('G', np.arange(30, 40), 3456.0, 'sol3')
        self.solK = CalSolution('K', np.arange(30, 40), 3456.0, 'solK')
        self.sol4 = CalSolution('G', np.arange(10), 4567.0, 'sol1')
        self.store = self.make_store('G')

    def test_latest_empty(self):
        self.assertIsNone(self.store.latest)

    def test_add_wrong_type(self):
        with self.assertRaises(ValueError):
            self.store.add(self.solK)

    def test_keep_latest(self):
        self.store.add(self.sol2)
        self.assertIs(self.store.latest, self.sol2)
        self.store.add(self.sol1)    # Earlier, should be ignored
        self.assertIs(self.store.latest, self.sol2)
        self.store.add(self.sol3)    # Later, should be kept
        self.assertIs(self.store.latest, self.sol3)

    def test_get_range(self):
        with self.assertRaises(NotImplementedError):
            self.store.get_range(1234.0, 2345.0)


class TestCalSolutionStore(TestCalSolutionStoreLatest):
    make_store = CalSolutionStore

    def test_get_range(self):
        self.store.add(self.sol2)
        self.store.add(self.sol1)
        self.store.add(self.sol3)
        soln = self.store.get_range(1234.0, 2345.0)
        self.assertEqual(soln.soltype, 'G')
        np.testing.assert_array_equal(soln.values, np.arange(20).reshape(2, 10))
        np.testing.assert_array_equal(soln.times, [1234.0, 2345.0])
        self.assertIsNone(soln.target)

    def test_get_range_with_target(self):
        self.store.add(self.sol1)
        self.store.add(self.sol3)
        self.store.add(self.sol4)
        solns = self.store.get_range(1234.0, 4567.0, target=self.sol1.target)
        self.assertEqual(solns.soltype, 'G')
        expect_sol1 = np.repeat(np.arange(10)[np.newaxis], 2, axis=0)
        np.testing.assert_array_equal(solns.values, expect_sol1)
        np.testing.assert_array_equal(solns.times, [1234.0, 4567.0])
        self.assertEqual(solns.target, self.sol1.target)
        # Get a nonexistent target
        solns = self.store.get_range(1234.0, 2345.0, target=self.sol2.target)
        self.assertEqual(solns.soltype, 'G')
        self.assertEqual(len(solns.values), 0)
        self.assertEqual(len(solns.times), 0)
        self.assertEqual(solns.target, self.sol2.target)
