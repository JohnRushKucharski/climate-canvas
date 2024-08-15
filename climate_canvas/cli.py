'''Starting place for command line application.'''

import typer
import numpy as np
import pandas as pd

from climate_canvas.plots_utilities import plot_response_surface

app  = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    '''Command line interface for climate_canvas.'''

@app.command()
def response(data: str = typer.Argument(..., help='Path to csv file.'),
             interpolate: bool = typer.Option(False, "--interp", help='Linearly interpolate data.'),
             title: str = typer.Option('Response Surface', help='Title of plot.'),
             xlabel: str = typer.Option('x', help='Label for x-axis.'),
             ylabel: str = typer.Option('y', help='Label for y-axis.'),
             zlabel: str = typer.Option('z', help='Label for z-axis.')):
    '''Plot response surface from climate impact data in csv file. In the form: \n
    \n

    |     | x0  | x1  | ... | xn  | \n
    |-----|-----|-----|-----|-----| \n
    | y0  | z00 | z01 | ... | Z0n | \n
    | y1  | z10 | .   |     | .   | \n
    | ... | .   |     |  .  | .   | \n
    | ... | .   |     |     | .   | \n
    | yn  | zn0 | ... |...  | znn | \n
    \n
    '''
    typer.echo('coming soon...')
    df = pd.read_csv(data)
    z = df.iloc[:, 1:].to_numpy(dtype=float)
    y = np.array(df.iloc[:, 0], dtype=float)
    x = np.array(df.columns.values[1:], dtype=float)

    plot_response_surface(x, y, z, interpolate)
