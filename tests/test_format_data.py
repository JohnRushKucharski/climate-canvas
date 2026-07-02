'''
Test laos style data formatting functions.
'''
from pathlib import Path
import unittest

from examples.multiple_response_surfaces.format_data import (is_valid_paths,
                                                             subplot_lists,
                                                             xy_levels,
                                                             format_data_files)

class TestFormatData(unittest.TestCase):
    '''Test laos style data formatting functions.'''

    def test_is_valid_path_bad_path_raises_error(self):
        '''Test format_data_files.'''
        paths = ['examples/multiple_response_surfaces/laos/x0_00_00.csv']
        with self.assertRaises(ValueError):
            format_data_files(paths, ['metric 2 (M2) [units/time]'])
    def test_is_valid_paths_good_path_returns_path(self):
        '''Test is_valid_path with good path.'''
        paths = ['examples/multiple_response_surfaces/laos/00_00_00.csv']
        self.assertEqual(is_valid_paths(paths), [Path(paths[0])])

    def test_subplot_lists(self):
        '''Test subplot_lists.'''	
        paths = list(Path('examples/multiple_response_surfaces/laos').glob('*.csv'))
        subplots = subplot_lists(paths)
        keys = ['-01', '00', '01']
        self.assertEqual(set(subplots.keys()), set(keys))
        self.assertEqual(len(subplots['00']), 6)
        self.assertEqual(len(subplots['01']), 6)
        self.assertEqual(len(subplots['-01']), 6)

    def test_xy_levels(self):
        '''Test xy_levels.'''
        paths = list(Path('examples/multiple_response_surfaces/laos').glob('*.csv'))
        subplots = subplot_lists(paths)
        rows, cols = xy_levels(subplots['00'])
        self.assertEqual(rows, [1, 0])
        self.assertEqual(cols, [-1, 0, 1])

    def test_format_data_files(self):
        '''Test format_data_files.'''
        paths = list(Path('examples/multiple_response_surfaces/laos').glob('*.csv'))
        format_data_files(paths, 'metric 2 (M2) [units/time]', 1)
        self.assertTrue(Path('examples/multiple_response_surfaces/example_metric_00.csv').exists())
