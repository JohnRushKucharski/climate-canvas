'''Tests for the data module.'''
import unittest

import numpy as np

from climate_canvas.data_utilities import (is_ascending, is_descending,
                                           is_interpolable, interpolate_2d,
                                           find_index, TruncateError, InterpolateError, Truncation,
                                           find_z, evenly_space)

class TestData(unittest.TestCase):
    #region is_ascending tests.
    '''Tests for the data module.'''
    def test_is_ascending_returns_true(self):
        '''Test is_ascending function.'''
        self.assertTrue(is_ascending(np.array([1, 2, 3, 4])))
    def test_is_ascending_returns_false(self):
        '''Test is_ascending function.'''
        self.assertFalse(is_ascending(np.array([1, 3, 2, 4])))
    def test_is_ascending_returns_true_for_empty_array(self):
        '''Test is_ascending function.'''
        self.assertTrue(is_ascending(np.array([])))
    def test_is_ascending_returns_true_for_single_element_array(self):
        '''Test is_ascending function.'''
        self.assertTrue(is_ascending(np.array([1])))
    def test_is_ascending_returns_true_for_inf_at_end(self):
        '''Test is_ascending function.'''
        self.assertTrue(is_ascending(np.array([1, 2, 3, np.inf])))
    def test_is_ascending_returns_false_with_nan_at_end(self):
        '''Test is_ascending function.'''
        self.assertFalse(is_ascending(np.array([1, 2, 3, np.nan])))
    def test_is_ascending_returns_false_with_nan_at_start(self):
        '''Test is_ascending function.'''
        self.assertFalse(is_ascending(np.array([np.nan, 2, 3, 4])))
    #endregion

    #region is_descending tests.
    def test_is_descending_returns_true(self):
        '''Test is_descending function.'''
        self.assertTrue(is_descending(np.array([4, 3, 2, 1])))
    def test_is_descending_returns_false(self):
        '''Test is_descending function.'''
        self.assertFalse(is_descending(np.array([1, 2, 3, 4])))
    def test_is_descending_returns_true_for_inf_at_start(self):
        '''Test is_descending function.'''
        self.assertTrue(is_descending(np.array([np.inf, 4, 3, 2, 1])))
    def test_is_descending_returns_true_for_neginf_at_end(self):
        '''Test is_descending function.'''
        self.assertTrue(is_descending(np.array([3, 2, 1, -np.inf])))
    #endregion

    #region is_interpolable tests.
    def test_is_interpolable_good_case_returns_true(self):
        '''Test is_interpolable function.'''
        self.assertTrue(is_interpolable(np.array([1, 2, 3, 4])))
    def test_is_interpolable_empty_returns_false(self):
        '''Test is_interpolable function.'''
        self.assertFalse(is_interpolable(np.array([])))
    def test_is_interpolable_single_element_returns_false(self):
        '''Test is_interpolable function.'''
        self.assertFalse(is_interpolable(np.array([1])))
    def test_is_interpolable_inf_returns_false(self):
        '''Test is_interpolable function.'''
        self.assertFalse(is_interpolable(np.array([1, 2, 3, np.inf])))
    def test_is_interpolable_nan_returns_false(self):
        '''Test is_interpolable function.'''
        self.assertFalse(is_interpolable(np.array([1, 2, 3, np.nan])))
    def test_is_interpolable_not_sorted_returns_false(self):
        '''Test is_interpolable function.'''
        self.assertFalse(is_interpolable(np.array([1, 3, 2, 4])))
    def test_is_interpolable_not_number_returns_false(self):
        '''Test is_interpolable function.'''
        self.assertFalse(is_interpolable(np.array([1, '2', 3, 4])))
    #endregion

    #region find_index tests.
    def test_find_index_in_set_returns_index(self):
        '''Test find_index function.'''
        self.assertEqual(find_index(2, np.array([1, 2, 3, 4])), (1,))
    def test_find_index_not_in_set_and_interpolate_is_true_returns_closest_indices(self):
        '''Test find_index function.'''
        self.assertEqual(find_index(2.5, np.array([1, 2, 3, 4]), interpolate=True), (1, 2))
    def test_find_index_lt_min_and_truncate_is_true_returns_0_false(self):
        '''Test find_index function.'''
        self.assertEqual(find_index(0, np.array([1, 2, 3, 4]), truncate=True), (0, False))
    def test_find_index_gt_max_and_truncate_is_true_returns_max_index_false(self):
        '''Test find_index function.'''
        self.assertEqual(find_index(5, np.array([1, 2, 3, 4]), truncate=True), (3, False))
    def test_find_index_not_in_set_and_interpolate_is_false_raises_interpolate_error(self):
        '''Test find_index function.'''
        with self.assertRaises(InterpolateError):
            find_index(2.5, np.array([1, 2, 3, 4]), interpolate=False)
    def test_find_index_lt_min_and_truncate_is_false_raises_truncate_error(self):
        '''Test find_index function.'''
        with self.assertRaises(TruncateError):
            find_index(0, np.array([1, 2, 3, 4]), truncate=False)
    def test_find_index_gt_max_and_truncate_is_false_raises_truncate_error(self):
        '''Test find_index function.'''
        with self.assertRaises(TruncateError):
            find_index(5, np.array([1, 2, 3, 4]), truncate=False)
    #endregion

    #region interpolate_2d tests.
    def test_interpolate_2d_returns_interpolated_value(self):
        '''Test interpolate_2d function.'''
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        xis, yis = (2, 3), (1, 2)
        self.assertEqual(interpolate_2d((xis, yis), (4.0, 6.5), xs, ys, zs), 0.5)
        self.assertEqual(interpolate_2d((xis, yis), (3.5, 6.5), xs, ys, zs), 0.25)
        self.assertEqual(interpolate_2d((xis, yis), (3.5, 7.0), xs, ys, zs), 0.5)
    #endregion

    #region: find z tests.
    #region: find z exact match tests.
    def test_find_z_returns_z_for_exact_match_square_grid(self):
        '''Test find_z function.'''
        xy = (3, 7) # exact match at index (2, 2).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs), 1)
    def test_find_z_returns_z_for_exact_match_non_square_grid(self):
        '''Test find_z function.'''
        xy = (3, 6) # exact match at index (2, 1).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs), 1)
    def test_find_z_returns_z_for_exact_match_at_min(self):
        '''Test find_z function.'''
        xy = (1, 5) # exact match at index (0, 0).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs), 1)
    def test_find_z_returns_z_for_exact_match_at_max(self):
        '''Test find_z function.'''
        xy = (4, 7) # exact match at index (3, 2).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(find_z(xy, xs, ys, zs), 1)
    #endregion

    #region: find z interpolation tests.
    def test_find_z_raises_interpolate_error_for_no_match_and_interpolate_is_false(self):
        '''Test find_z function.'''
        xy = (2.5, 2.5) # no match.
        xs, ys = (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        with self.assertRaises(InterpolateError):
            find_z(xy, xs, ys, zs, interpolate=False)

    #region: find z 1D interpolation tests.
    def test_find_z_interpolation_is_true_returns_interpolated_value_exact_y(self):
        '''Test find_z function.'''
        xy = (2., 2.5) # interpolate between (2, 2) and (2, 3).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        zs = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), 1.5)
    def test_find_z_interpolation_is_true_returns_interpolated_value_exact_x(self):
        '''Test find_z function.'''
        xy = (2.5, 2.)
        xs, ys = (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        zs = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), 1.5)
    def test_find_z_1d_50p_downward_interpolation_on_x_axis(self):
        '''Test find_z function.'''
        xy = (2.5, 6) # interpolate between (2, 1) and (2, 2).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), 0.5)
    def test_find_z_1d_50p_upward_interpolation_on_x_axis(self):
        '''Test find_z function.'''
        xy = (2.5, 6) # interpolate between (2, 2) and (2, 3).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), 0.5)
    def test_find_z_1d_30p_downward_interpolation_on_x_axis(self):
        '''Test find_z function.'''
        xy = (2.3, 6) # interpolate between (2, 1) and (2, 2).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(find_z(xy, xs, ys, zs, interpolate=True),
                                0.7) # 30% of distance between 1 and 0.
    def test_find_z_1d_30p_upward_interpolation_on_x_axis(self):
        '''Test find_z function.'''
        xy = (2.3, 6) # interpolate between (2, 2) and (2, 3).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(find_z(xy, xs, ys, zs, interpolate=True),
                                0.3) # 30% of distance between 0 and 1.
    def test_find_z_1d_50p_interpolation_on_x_axis_zgrid_crosses0(self):
        '''Test find_z function.'''
        xy = (2.5, 6)
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, -4, 2, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), -1)
    def test_find_z_1d_50p_interpolation_on_x_axis_neg_zgrid(self):
        '''Test find_z function.'''
        xy = (2.5, 6)
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, -3, -1, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), -2)
    def test_find_z_1d_50p_interpolation_on_neg_x_axis(self):
        '''Test find_z function.'''
        xy = (-2.5, 6)
        xs, ys = (np.array([-4, -3, -2, -1]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 1, 3, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), 2)
    # only one y axis test is needed as the interpolation is the same for x and y.
    def test_find_z_1d_50p_interpolation_on_y_axis(self):
        '''Test find_z function.'''
        xy = (2, 5.5)
        xs, ys = (np.array([1, 2, 3]), np.array([4, 5, 6, 7]))
        zs = np.array([[0, 0, 0], [0, 1, 0], [0, 3, 0], [0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, interpolate=True), 2)
    #endregion

    #region: find z 2d interpolation tests
    def test_find_z_2d_50p_interpolation(self):
        '''Test find_z function.'''
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(find_z((4, 6.5), xs, ys, zs, interpolate=True), 0.5) # 1D
        self.assertEqual(find_z((3.5, 7), xs, ys, zs, interpolate=True), 0.5) # 1D
        self.assertEqual(find_z((3.5, 6.5), xs, ys, zs, interpolate=True), 0.25) # 2D
    def test_find_z_interpolation_is_true_returns_interpolated_value_between_x_and_y(self):
        '''Test find_z function.'''
        #xy = (1.5, 1.5) # interpolate between (2, 2) and (3, 3).
        xs, ys = (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        zs = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z((1.5, 1.5), xs, ys, zs, interpolate=True), 0.25) # 0.25 = 0.5 * 0.5. Less obvious result for 2D interpolation. # pylint: disable=line-too-long
        self.assertEqual(find_z((2.5, 2.5), xs, ys, zs, interpolate=True), 1.75) # 1.75 = 1.5 + 0.5 * 0.5. # pylint: disable=line-too-long
    #endregion

    #region: find z truncation tests.
    def test_find_z_raises_truncate_error_for_no_match_and_truncation_is_none(self):
        '''Test find_z function.'''
        xy = (0, 0)
        xs, ys = (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        with self.assertRaises(TruncateError):
            find_z(xy, xs, ys, zs, truncation=Truncation.NONE)
    def test_find_z_with_x_lt_min_truncation_edge_returns_edge_value(self):
        '''Test find_z function.'''
        xy = (0, 6)
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, truncation=Truncation.EDGE), 1)
    def test_find_z_with_x_gt_max_truncation_edge_returns_edge_value(self):
        '''Test find_z function.'''
        xy = (5, 6)
        xs, ys = (np.array([1, 2, 3, 4]), np.array([5, 6, 7]))
        zs = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, truncation=Truncation.EDGE), 1)
    def test_find_z_with_y_lt_min_truncation_edge_returns_edge_value(self):
        '''Test find_z function.'''
        xy = (2, 3)
        xs, ys = (np.array([1, 2, 3]), np.array([4, 5, 6, 7]))
        zs = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, truncation=Truncation.EDGE), 1)
    def test_find_z_with_y_gt_max_truncation_edge_returns_edge_value(self):
        '''Test find_z function.'''
        xy = (2, 8)
        xs, ys = (np.array([1, 2, 3]), np.array([4, 5, 6, 7]))
        zs = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]])
        self.assertEqual(find_z(xy, xs, ys, zs, truncation=Truncation.EDGE), 1)
    def test_find_corner_cases_truncation_edge_returns_edge_value(self):
        '''Test find_z function.'''
        xs, ys = (np.array([1, 2, 3]), np.array([4, 5, 6, 7]))
        zs_ul = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(find_z((0, 3), xs, ys, zs_ul, truncation=Truncation.EDGE), 1)
        zs_ur = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(find_z((4, 3), xs, ys, zs_ur, truncation=Truncation.EDGE), 1)
        zs_ll = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]])
        self.assertEqual(find_z((0, 8), xs, ys, zs_ll, truncation=Truncation.EDGE), 1)
        zs_lr = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(find_z((4, 8), xs, ys, zs_lr, truncation=Truncation.EDGE), 1)
    #endregion
    #endregion
    #endregion

    #region: evenly_space tests.
    def test_evenly_space_deltas_returns_evenly_spaced_arrays(self):
        '''Test evenly_space function.'''
        xs = np.array([1, 2])
        ys = np.array([3, 4])
        zs = np.array([[0, 0], [0, 1]])
        x, y, z = evenly_space(xs, ys, zs, (0.5, 0.5), None)
        self.assertTrue(np.allclose(x, np.array([1, 1.5, 2])))
        self.assertTrue(np.allclose(y, np.array([3, 3.5, 4])))
        self.assertTrue(np.allclose(z, np.array([[0, 0, 0], [0, 0.25, 0.50], [0, 0.50, 1]])))

    def test_evenly_space_deltas_returns_evenly_spaced_for_unequal_x_y_lens(self):
        '''Test evenly_space function.'''
        xs = np.array([1, 2, 3])
        ys = np.array([4, 5])
        zs = np.array([[0, 0, 0], [0, 0, 1]])
        x, y, z = evenly_space(xs, ys, zs, (0.5, 0.5), None)
        self.assertTrue(np.allclose(x, np.array([1, 1.5, 2, 2.5, 3])))
        self.assertTrue(np.allclose(y, np.array([4,4.5, 5])))
        self.assertTrue(np.allclose(
            z, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0.25, 0.50], [0, 0, 0, 0.50, 1]])))

    def test_evenly_space_increments_returns_evenly_spaced_arrays(self):
        '''Test evenly_space function.'''
        xs = np.array([1, 2])
        ys = np.array([3, 4])
        zs = np.array([[0, 0], [0, 1]])
        x, y, z = evenly_space(xs, ys, zs, None, (3,3))
        self.assertTrue(np.allclose(x, np.array([1, 1.5, 2])))
        self.assertTrue(np.allclose(y, np.array([3, 3.5, 4])))
        self.assertTrue(np.allclose(z, np.array([[0, 0, 0], [0, 0.25, 0.50], [0, 0.50, 1]])))

    def test_evenly_space_increments_returns_evenly_spaced_for_unequal_x_y_lens(self):
        '''Test evenly_space function.'''
        xs = np.array([1, 2, 3])
        ys = np.array([4, 5])
        zs = np.array([[0, 0, 0], [0, 0, 1]])
        x, y, z = evenly_space(xs, ys, zs, None, (5, 3))
        self.assertTrue(np.allclose(x, np.array([1, 1.5, 2, 2.5, 3])))
        self.assertTrue(np.allclose(y, np.array([4, 4.5, 5])))
        self.assertTrue(np.allclose(
            z, np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0.25, 0.50], [0, 0, 0, 0.50, 1]])))
    #endregion
