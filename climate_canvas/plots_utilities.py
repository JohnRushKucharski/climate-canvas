'''Plotting utilities for climate impact data.'''
from matplotlib import pyplot as plt

from climate_canvas.data_utilities import evenly_space

def plot_response_surface(xs, ys, zs,
                          interpolate: bool = False,
                          labels: tuple[str, str, str] = ('x', 'y', 'z'),
                          title: str = 'Response Surface'
                          ) -> None:
    '''Plot response surface from climate impact data.'''
    fig, ax = plt.subplots(figsize=(10, 10))
    if interpolate: # and
        xs, ys, zs = evenly_space(xs, ys, zs, None, (25, 25))
    extent = (xs.min(), xs.max(), ys.min(), ys.max())
    im = ax.imshow(zs, extent=extent, aspect='auto', cmap='RdYlBu')
    # todo: contour lines are rotated, this has happened to others.
    ax.contour(zs, extent=extent, origin='image', colors='black')
    fig.colorbar(im, ax=ax)
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    plt.show()

    #     im = ax.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()))
    #     ax.contour(z, colors='black')
    #     fig.colorbar(im, ax=ax)
    #     plt.show()
    # else:
    #     Z = zs
    #     X, Y = np.meshgrid(xs, ys)
    #     #fig, ax = plt.subplots()
    #     im = ax.imshow(Z)
    #     ax.contour(Z, colors='black')
    #     fig.colorbar(im, ax=ax)
    #     plt.show()

    # print(X.shape)
    # print(Z.shape)

    # print(ys)
    # print(xs)
    # print(zs)
    # for i, row in enumerate(Y): # i.e., temp
    #     for j, col in enumerate(X): # i.e., precip
    #         Z[i, j] = find_z((col[i], row[j]), (xs, ys), zs)
    #         print(f'x/p:{col[i]}, y/t:{row[j]}, z:{Z[i, j]}')
    # print(Z.shape)
    # print(Z)

    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)
    # print(Z)
    # print(X.shape)
    # print(Y.shape)
    # Z1 = np.exp(-X**2 - Y**2)
    # Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    # Z = (Z1 - Z2) * 2
    # print(Z1.shape)
    # print(Z2.shape)
    # print(Z.shape)
    # print(Z)
