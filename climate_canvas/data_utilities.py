'''Data analysis functions for climate_canvas'''
from enum import Enum
from typing import Callable

import numpy as np

def is_ascending(array: np.ndarray) -> bool:
    '''Check if array is sorted in ascending order.'''
    return bool(np.all(array[:-1] <= array[1:]))

def is_descending(array: np.ndarray) -> bool:
    '''Check if array is sorted in descending order.'''
    return bool(np.all(array[:-1] >= array[1:]))

def is_interpolable(array: np.ndarray) -> bool:
    '''Check if array of data is interpolable.'''
    if len(array) < 2:
        return False
    if not np.issubdtype(array.dtype, np.number):
        return False
    if not is_ascending(array) and not is_descending(array):
        return False
    return not np.isinf(np.diff(array)).any() # False if inf exist, True otherwise.

def interpolate_2d(indices: tuple[tuple[int, int], tuple[int, int]], xy: tuple[float, float],
                   xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> float:
    '''
    Bilinear interpolation on 2D grid.
    Finds z value at (x, y) in on grid of zs in (xs, ys).

    Args:
        indices: tuple of x (col) and y (row) indices.
        xy: point to find on 2D plane, tuple of x and y values.
        xs: column values.
        ys: row values.
        zs: 2D grid values.
    
    Returns:
        z value at (x, y).

    Note: 
        This is a simple implementation of SciPy.interpolate.RegularGridInterpolator.
    '''
    xi0, xi1 = indices[0]
    yi0, yi1 = indices[1]
    x0, x1 = xs[xi0], xs[xi1]
    y0, y1 = ys[yi0], ys[yi1]
    z00, z01 = zs[yi0, xi0], zs[yi0, xi1]
    z10, z11 = zs[yi1, xi0], zs[yi1, xi1]
    # Structure of reduced 2D plane:
    #
    #    | x0  x1
    # ------------
    # y0 | z00 z01
    # y1 | z10 z11
    #
    # Use px to find z-value:
    # @ points: (x,0), (x, 1)
    # e.g. zx0, zx1 in diagram.
    # Use py to find z-value:
    # @ points: (x, y) i.e zxy.
    #
    #    | x0   x   x1
    # ----------v------
    # y0 | z00 zx0  z01 |
    # y------->zxy      + py
    # y1 | z10 zx1  z11
    #      .----+
    #        px
    #
    # x portion of x0 - x1 line segment
    px = (xy[0] - x0) / (x1 - x0)
    # z-value at (x, y0)
    zx0 = z00 * (1 - px) + z01 * px
    # z-value at (x, y1)
    zx1 = z10 * (1 - px) + z11 * px
    # y portion of y0 - y1 line segment
    py = (xy[1] - y0) / (y1 - y0)
    # z-value at (x, y)
    zxy = zx0 * (1 - py) + zx1 * py
    return zxy

    # Earlier attempt.
    # percent change in xs.
    # px = (xy[0] - x0) / (x1 - x0)
    # # change in z along rows.
    # dzdx_y0 = (z01 - z00) * px
    # dzdx_y1 = (z11 - z10) * px
    # # percent change in ys
    # py = (xy[1] - y0) / (y1 - y0)
    # # change in z along cols.
    # dzdy_x0 = (z10 - z00) * py
    # dzdy_x1 = (z11 - z01) * py
    # # change in z along x and y axes.
    # dzdx = dzdx_y0 * py + dzdx_y1 * (1 - py)
    # dzdy = dzdy_x0 * px + dzdy_x1 * (1 - px)
    # # interpolate z value at (x, y).
    # return z00 + dzdx + dzdy

class TruncateError(Exception):
    '''Raised when value is not on domain of set and truncate is False.'''
class InterpolateError(Exception):
    '''Raised when value is not in set and interpolate is False.'''

Truncation = Enum('Truncation', ['NONE', 'NAN', 'EDGE'])

def z_truncate(row: int, col: int, zs: np.ndarray,
             truncation: Truncation) -> float:
    '''Truncate index based on truncation method.'''
    match truncation:
        case Truncation.NONE:
            raise TruncateError('Value is not on domain of set.')
        case Truncation.NAN:
            return np.nan
        case Truncation.EDGE:
            return zs[row, col]
        case _:
            raise NotImplementedError(f'{truncation} is not a valid truncation method.')

def find_index(value: float, array: np.ndarray,
               interpolate: bool = False, truncate: bool = False
               ) -> tuple[int|bool,...]:
    '''Find index of value in set.
    
    Args:
        value: value to find.
        array: set of values.
        interpolate: provides indices for closest values if True.
            If False, an interpolate error is raised when no exact match is found.
        truncate: provides indices for values at the ends of the set if True.
            If False, a truncate error is raised when value is not in the set.
    
    Returns:
        index of value in set.
        if value is not in set and interpolate is True, returns indices of closest values.
        if value not on domain of set & truncate is True, returns tuple: (index at edge, False).

    Raises:
        ValueError: if value is not in set and array is not interpolable.
        InterpolateError: if value is not in set and interpolate is False.
        TruncateError: if value is not in domain of set and truncate is False.
    '''
    if value in array:
        return (np.where(array == value)[0][0],) # first occurence.
    if not is_interpolable(array):
        raise ValueError(f'''Interpolate and/or truncate not possible.
                         The array {array} must be numeric, sorted, and contain no nan values.''')
    if is_ascending(array):
        if value < array[0]:
            if not truncate:
                raise TruncateError(f'{value} not on domain of {array}. Use truncate=True.')
            return (0, False)
        if value > array[-1]:
            if not truncate:
                raise TruncateError(f'{value} not on domain of {array}. Use truncate=True.')
            return (len(array) - 1, False)
        # find closest pair of values.
        if not interpolate:
            raise InterpolateError(f'{value} not in {array}. Use interpolate=True.')
        for i, val in enumerate(array):
            if value < val:
                return (i-1, i)
    else: # descending array.
        if value > array[0]:
            if not truncate:
                raise TruncateError(f'{value} not on domain of {array}. Use truncate=True.')
            return (0, False)
        if value < array[-1]:
            if not truncate:
                raise TruncateError(f'{value} not on domain of {array}. Use truncate=True.')
            return (len(array) - 1, False)
        # find closest pair of values.
        if not interpolate:
            raise InterpolateError(f'{value} not in {array}. Use interpolate=True.')
        for i, val in enumerate(array):
            if value > val:
                return (i-1, i)
    raise ValueError(f'Something went wrong finding value {value} in {array}.')
    # # find closest pair of values.
    # if not interpolate:
    #     raise InterpolateError(f'{value} not in {array}. Use interpolate=True.')
    # else:
    #     for i, val in enumerate(array):
    #         if value < val:
    #             return (i-1, i)
    # # for i, val in enumerate(array):
    # #     if value < val:
    #         if i == 0: # value is less than min.
    #             if not truncate:
    #                 raise TruncateError(f'{value} not on domain of {array}. Use truncate=True.')
    #             return (0, False)
    #         if not interpolate:
    #             raise InterpolateError(f'{value} not in {array}. Use interpolate=True.')
    #         return (i-1, i)
    # if not truncate: # value is greater than max.
    #     raise TruncateError(f'{value} not on domain of {array}. Use truncate=True.')
    # return (len(array) - 1, False)

def find_z(xy: tuple[float, float],
           xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
           interpolate: bool = False, truncation: Truncation = Truncation.NONE
           )-> float:
    '''Find z value at (x, y) in (xs, ys).
    
    Args:
        xy: tuple of x and y values.
        xs: column values.
        ys: row values.
        zs: 2D grid values.
        interpolate: linear interpolation of z values not in zs if True.
            A value error is raised if False and (x, y) is not in (xs, ys).
        truncation: method to handle values not on domain of (xs, ys).
            NONE: raise TruncateError.
            NAN: return np.nan.
            EDGE: return value at edge of domain.

    Returns:
        z value at (x, y).

    Raises:
        InterpolateError: if (x, y) is not in (xs, ys) and interpolate is False.
        TruncateError: if (x, y) is not on the domains of (xs, ys) and truncation is NONE.
    '''
    _truncate = False if truncation == Truncation.NONE else True
    col = find_index(xy[0], xs, interpolate, _truncate)
    row = find_index(xy[1], ys, interpolate, _truncate)
    match len(row): # y-axis.
        case 1: #  fixed row.
            match len(col): # x-axis.
                case 1:
                    # exact match z[y-index, x-index].
                    # fixed row (y-axis) and col (x-axis).
                    return zs[row[0], col[0]]
                case 2:
                    # truncated col (x-axis).
                    if col[1] is False:
                        # row (y-axis) is exact match.
                        # col (x-axis) is truncated value.
                        return z_truncate(row[0], col[0], zs, truncation)
                    # 1D interpolation on col (x-axis).
                    else:
                        z0 = zs[row[0], col[0]]
                        z1 = zs[row[0], col[1]]
                        px = (xy[0] - xs[col[0]]) / (xs[col[1]] - xs[col[0]])
                        return z0 + px * (z1 - z0)
                case _:
                    # something went wrong in find_index().
                    raise ValueError('xi must have length 1 or 2.')
        # truncated or interpolated row values.
        case 2: # y-axis (row).
            match len(col):
                case 1:
                    # truncated row (y-axis).
                    if row[1] is False:
                        # row (y-axis) is truncated value.
                        # col (x-axis) is exact match.
                        return z_truncate(row[0], col[0], zs, truncation)
                    # 1D interpolation on row (y-axis)
                    else:
                        z0 = zs[row[0], col[0]]
                        z1 = zs[row[1], col[0]]
                        py = (xy[1] - ys[row[0]]) / (ys[row[1]] - ys[row[0]])
                        return z0 + py * (z1 - z0)
                case 2:
                    # corner case: row & col are truncated.
                    if col[1] is False and row[1] is False:
                        # both row and col are truncated values.
                        return z_truncate(row[0], col[0], zs, truncation)
                    # 2D interpolation on plane.
                    if not (len(col) == 2 and len(row) == 2 and
                            all(isinstance(i, int) for i in row) and
                            all(isinstance(i, int) for i in col)):
                        raise ValueError('Interpolation on 2D plane requires 2x2 grid of indices.')
                    return interpolate_2d((col, row), xy, xs, ys, zs)
                case _:
                    # something went wrong in find_index().
                    raise ValueError('xi must have length 1 or 2.')
        case _:
            # something went wrong in find_index().
            raise ValueError('yi must have length 1 or 2.')

def interpolator(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                 truncation_method: Truncation = Truncation.NONE
                 )-> Callable[[tuple[float, float]], float]:
    '''
    Create bilinear interpolator for data on 2D grid.
    
    note: This is a simple implementation of SciPy.interpolate.RegularGridInterpolator.
    '''
    def interpolate(xy: tuple[float, float]) -> float:
        '''Interpolate z value at (x, y) in (xs, ys).'''
        return find_z(xy, xs, ys, zs, interpolate=True,
                      truncation=truncation_method)
    return interpolate

def evenly_space(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                 deltas: None|tuple[float, float], increments: None|tuple[int, int],
                 truncation_method: Truncation = Truncation.NONE
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Evenly space impact assessment x, y, z data.'''
    # xs = np.array(df.iloc[:, 0].values, dtype=float)
    # ys = np.array(df.columns.values[1:], dtype=float)
    if not is_interpolable(xs) or not is_interpolable(ys):
        raise ValueError('xs or ys are not interpolable.')
    if deltas is not None and increments is not None:
        raise ValueError('Both deltas and increments cannot be provided.')
    if deltas is None and increments is None:
        return (xs, ys, zs) # no delta or increment provided.

    f = interpolator(xs, ys, zs, truncation_method)
    #interpolator = RegularGridInterpolator((xs, ys), zs, method='linear')
    if increments is not None and deltas is None:
        new_xs = np.linspace(xs[0], xs[-1], increments[0])
        new_ys = np.linspace(ys[0], ys[-1], increments[1])

    if deltas is not None and increments is None:
        new_xs = []
        x = xs.min()
        while x < xs.max():
            new_xs.append(x)
            x += deltas[0]
        new_xs.append(xs.max())
        new_xs = np.array(new_xs, dtype=float)

        new_ys = []
        y = ys.min()
        while y < ys.max():
            new_ys.append(y)
            y += deltas[1]
        new_ys.append(ys.max())
        new_ys = np.array(new_ys, dtype=float)

    new_zs = np.empty((len(new_ys), len(new_xs)), dtype=float)
    for i, y in enumerate(new_ys): # rows
        for j, x in enumerate(new_xs): # columns
            new_zs[i, j] = f((x, y))
    return (new_xs, new_ys, new_zs)
