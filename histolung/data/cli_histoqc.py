import click
import logging
from pathlib import Path
from histolung.data.histoqc import run_histoqc_raw_path_mounted  # Update with correct import path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_dir',
                type=click.Path(exists=True,
                                file_okay=False,
                                dir_okay=True,
                                readable=True,
                                resolve_path=True))
@click.argument('output_dir',
                type=click.Path(file_okay=False,
                                dir_okay=True,
                                writable=True,
                                resolve_path=True))
@click.argument('config_path',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True,
                                resolve_path=True))
@click.option('--input-pattern',
              default="*.svs",
              help="File pattern to match within input_dir (default: *.svs).")
@click.option('--docker-image',
              default="histotools/histoqc:master",
              help="Docker image for HistoQC.")
@click.option(
    '--user',
    default=None,
    help="User identifier for Docker container (default: current user).")
@click.option('--force',
              is_flag=True,
              help="Force processing even if outputs exist.")
@click.option('--num-workers',
              type=int,
              default=None,
              help="Number of parallel workers.")
def cli(input_dir, output_dir, config_path, input_pattern, docker_image, user,
        force, num_workers):
    """
    CLI to run HistoQC using the raw path mounted method.
    """
    try:
        run_histoqc_raw_path_mounted(input_dir=Path(input_dir),
                                     output_dir=Path(output_dir),
                                     input_pattern=input_pattern,
                                     docker_image=docker_image,
                                     user=user,
                                     config_path=Path(config_path),
                                     force=force,
                                     num_workers=num_workers)
        logger.info("HistoQC processing completed successfully.")
    except Exception as e:
        logger.error(f"Error running HistoQC: {e}")


if __name__ == '__main__':
    cli()
