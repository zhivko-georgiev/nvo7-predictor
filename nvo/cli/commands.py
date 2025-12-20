"""CLI commands for NVO Rankings."""
import click
import pandas as pd

from nvo.config import load_config
from nvo.utils.logger import setup_logger, get_logger
from nvo.services import run_predictions, run_validation, run_analysis
from nvo.display import (
    save_results,
    format_prediction_metrics,
    format_validation_metrics,
    format_top_predictions,
    format_worst_predictions,
    format_yearly_stats,
    format_trends
)

logger = get_logger("cli")


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.option('--verbose', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config, verbose):
    """NVO 7th Grade Rankings Prediction System."""
    ctx.ensure_object(dict)
    cfg = load_config(config)
    ctx.obj['config'] = cfg
    
    log_level = 'DEBUG' if verbose else cfg.logging.get('level', 'INFO')
    log_file = cfg.logging.get('file_path') if cfg.logging.get('file') else None
    setup_logger('nvo', log_level, cfg.logging.get('console', True), log_file)


@cli.command()
@click.option('--year', type=int, help='Year to predict')
@click.option('--gender', type=click.Choice(['male', 'female', 'all']), help='Gender filter')
@click.option('--schools', help='Comma-separated school names to filter')
@click.option('--no-cache', is_flag=True, help='Force retraining (ignore cached models)')
@click.pass_context
def predict(ctx, year, gender, schools, no_cache):
    """Generate predictions with confidence intervals."""
    cfg = ctx.obj['config']
    
    predict_year = year or cfg.data['predict_year']
    gender_filter = None if gender == 'all' else (gender or cfg.filters.get('gender'))
    school_filter = schools.split(',') if schools else cfg.filters.get('schools', [])
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Predicting for {predict_year}")
    click.echo(f"Training on: {cfg.data['historical_years']}")
    click.echo('='*60)
    
    results, metrics = run_predictions(
        cfg.data['historical_years'], predict_year, cfg.data['files_dir'],
        cfg.model, gender_filter, school_filter or None, use_cache=not no_cache
    )
    
    if not results:
        click.echo("No predictions generated")
        return
    
    for line in format_prediction_metrics(metrics):
        click.echo(f"\n{line}")
    
    results_df, output_file = save_results(
        results, cfg.data['output_dir'],
        f"Predictions_{predict_year}_{gender_filter or 'all'}.xlsx"
    )
    
    click.echo("\n=== Top 10 Predictions ===")
    click.echo(format_top_predictions(results_df, gender_filter))
    click.echo(f"\n✓ Full results saved to {output_file}")


@cli.command()
@click.option('--test-year', type=int, required=True, help='Year to validate against')
@click.option('--gender', type=click.Choice(['male', 'female', 'all']), default='female', help='Gender filter')
@click.option('--schools', help='Comma-separated school names to filter')
@click.pass_context
def validate(ctx, test_year, gender, schools):
    """Validate model against actual results."""
    cfg = ctx.obj['config']
    
    train_years = [y for y in cfg.data['historical_years'] if y < test_year]
    if len(train_years) < 2:
        click.echo(f"Error: Need at least 2 years before {test_year} for training")
        return
    
    gender_filter = None if gender == 'all' else gender
    school_filter = schools.split(',') if schools else None
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Validating against {test_year}")
    click.echo(f"Training on: {train_years}")
    click.echo('='*60)
    
    results, metrics = run_validation(
        train_years, test_year, cfg.data['files_dir'],
        cfg.model, gender_filter, school_filter
    )
    
    if not results:
        click.echo("No results generated")
        return
    
    for line in format_validation_metrics(metrics):
        click.echo(line)
    
    results_df, output_file = save_results(
        results, cfg.data['output_dir'],
        f"Validation_{test_year}_{gender_filter or 'all'}.xlsx"
    )
    
    click.echo("\n=== Top 10 Worst Predictions (R1) ===")
    click.echo(format_worst_predictions(results_df, gender_filter))
    click.echo(f"\n✓ Detailed results saved to {output_file}")


@cli.command()
@click.option('--years', help='Comma-separated years to analyze')
@click.option('--gender', type=click.Choice(['male', 'female', 'all']))
@click.option('--schools', help='Comma-separated school names to filter')
@click.pass_context
def analyze(ctx, years, gender, schools):
    """Analyze historical rankings data."""
    cfg = ctx.obj['config']
    
    years_list = [int(y) for y in years.split(',')] if years else cfg.data['historical_years'][-2:]
    gender_filter = None if gender == 'all' else (gender or cfg.filters.get('gender'))
    school_filter = schools.split(',') if schools else None
    target = 'Female' if gender_filter and gender_filter.lower().startswith('f') else \
             'Male' if gender_filter and gender_filter.lower().startswith('m') else 'Total'
    
    logger.info(f"Analyzing years: {years_list}")
    
    yearly_stats, trends = run_analysis(years_list, cfg.data['files_dir'], gender_filter, school_filter)
    
    for line in format_yearly_stats(yearly_stats):
        click.echo(line)
    
    if school_filter and len(years_list) > 1 and trends:
        for line in format_trends(trends, target):
            click.echo(line)


if __name__ == '__main__':
    cli()
