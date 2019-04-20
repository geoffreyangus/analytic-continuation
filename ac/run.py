"""
"""
from pet_ct.util.util import log_title, load_params, extract_kwargs

import click

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.argument(
    "process_dir",
    type=str,
)
@click.option(
    "--overwrite",
    is_flag=True
)
@click.option(
    "--notify",
    type=bool,
    default=True
)
@click.option(
    "--commit/--no-commit",
    default=False
)
@click.argument(
    "kwargs",
    nargs=-1,
    type=click.UNPROCESSED
)
def run(process_dir, overwrite, notify, commit, kwargs):
    """
    """
    kwargs = extract_kwargs(kwargs)
    params = load_params(process_dir)

    process_class = params["process_class"]
    log_title(process_class)

    process = globals()[process_class](process_dir, **params["process_args"])
    process.run(overwrite=overwrite, notify=notify, commit=commit, **kwargs)
