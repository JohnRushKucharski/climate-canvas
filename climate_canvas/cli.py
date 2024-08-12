'''Starting place for command line application.'''

import typer

app  = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    '''Command line interface for climate_canvas.'''

@app.command()
def response():
    '''Plot response surface from csv file.'''
    typer.echo('coming soon...')
