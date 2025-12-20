"""CLI commands for NVO Rankings."""
import click
import pandas as pd
import numpy as np
from pathlib import Path
from nvo.config import load_config
from nvo.utils.logger import setup_logger, get_logger
from nvo.data.processors import build_dataset
from nvo.data.exam_loaders import load_exam_distribution
from nvo.models.trainer import train_model, prepare_features
from nvo.models.prediction_utils import (
    prepare_prediction_data,
    generate_predictions,
    compute_prediction_intervals,
    compute_metrics,
)

logger = get_logger("cli")


def get_gender_list(gender_filter: str) -> list:
    """Get list of genders to process based on filter."""
    if gender_filter and gender_filter.lower().startswith('f'):
        return ['Female']
    elif gender_filter and gender_filter.lower().startswith('m'):
        return ['Male']
    return ['Total', 'Male', 'Female']


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
@click.pass_context
def predict(ctx, year, gender, schools):
    """Generate predictions with confidence intervals."""
    cfg = ctx.obj['config']
    
    predict_year = year or cfg.data['predict_year']
    gender_filter = None if gender == 'all' else (gender or cfg.filters.get('gender'))
    school_filter = schools.split(',') if schools else cfg.filters.get('schools', [])
    
    historical_years = cfg.data['historical_years']
    template_year = max(historical_years)
    prev_year = template_year
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Predicting for {predict_year}")
    click.echo(f"Training on: {historical_years}")
    click.echo('='*60)
    
    # Load data
    df_hist = build_dataset(historical_years, cfg.data['files_dir'])
    df_template = df_hist[df_hist['Year'] == template_year].copy()
    
    # Load exam features for prediction year
    exam_features = load_exam_distribution(predict_year, cfg.data['files_dir'])
    if exam_features:
        logger.info(f"Loaded {len(exam_features)} exam features for {predict_year}")
        for key, val in exam_features.items():
            df_template[key] = val
    
    if school_filter:
        mask = df_template['School'].str.contains('|'.join(school_filter), case=False, na=False)
        df_template = df_template[mask]
    
    click.echo(f"\nTraining samples: {len(df_hist)}")
    click.echo(f"Prediction targets: {len(df_template)}\n")
    
    all_results = {}
    genders = get_gender_list(gender_filter)
    
    for round_num in [1, 2]:
        for g in genders:
            target_col = f'R{round_num}_Min_{g}'
            if target_col not in df_hist.columns:
                continue
            
            click.echo(f"Training model for R{round_num} {g}...")
            
            # Train model
            model, le_school, le_profile, feature_cols, school_stats = train_model(
                df_hist, target_col, round_num, cfg.model
            )
            
            if model is None:
                continue
            
            avg_volatility = np.mean([s['volatility'] for s in school_stats.values()])
            
            # Prepare prediction data
            X, df_prep, prev_scores_map, mask = prepare_prediction_data(
                df_template, df_hist, target_col, prev_year,
                le_school, le_profile, school_stats, feature_cols
            )
            
            # Generate predictions
            results = generate_predictions(
                model, X, df_prep, prev_scores_map, school_stats,
                pd.Series([True] * len(X), index=X.index)  # All rows valid for prediction
            )
            
            # Store results with intervals
            for r in results:
                key = (r.school, r.profile)
                if key not in all_results:
                    all_results[key] = {'School': r.school, 'Profile': r.profile}
                
                lower, upper, confidence = compute_prediction_intervals(
                    r.predicted, r.volatility, avg_volatility
                )
                
                all_results[key][f'R{round_num}_{g}_Predicted'] = round(r.predicted, 2)
                all_results[key][f'R{round_num}_{g}_Lower'] = round(lower, 2)
                all_results[key][f'R{round_num}_{g}_Upper'] = round(upper, 2)
                all_results[key][f'R{round_num}_{g}_Confidence'] = round(confidence, 1)
                all_results[key][f'R{round_num}_{g}_Volatility'] = round(r.volatility, 1)
                all_results[key][f'R{round_num}_{g}_Years_Data'] = r.n_years
                all_results[key][f'R{round_num}_{g}_Reliable'] = r.reliable
            
            reliable_count = sum(1 for r in results if r.reliable)
            click.echo(f"  Predictions: {len(results)}, Reliable: {reliable_count}\n")
    
    if not all_results:
        click.echo("No predictions generated")
        return
    
    # Save results
    results_df = pd.DataFrame(list(all_results.values()))
    
    # Sort by predicted score
    sort_col = [c for c in results_df.columns if c.endswith('_Predicted')]
    if sort_col:
        results_df = results_df.sort_values(sort_col[0], ascending=False)
    
    output_dir = Path(cfg.data['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    gender_suffix = gender_filter if gender_filter else "all"
    output_file = output_dir / f"Predictions_{predict_year}_{gender_suffix}.xlsx"
    results_df.to_excel(output_file, index=False)
    
    # Display top predictions
    display_cols = ['School', 'Profile']
    target = genders[0] if len(genders) == 1 else 'Female'
    display_cols.extend([
        f'R1_{target}_Predicted', f'R1_{target}_Lower', 
        f'R1_{target}_Upper', f'R1_{target}_Confidence'
    ])
    display_cols = [c for c in display_cols if c in results_df.columns]
    
    click.echo("=== Top 10 Predictions ===")
    click.echo(results_df[display_cols].head(10).to_string(index=False))
    click.echo(f"\n✓ Full results saved to {output_file}")


@cli.command()
@click.option('--test-year', type=int, required=True, help='Year to validate against')
@click.option('--gender', type=click.Choice(['male', 'female', 'all']), default='female', help='Gender filter')
@click.option('--schools', help='Comma-separated school names to filter')
@click.pass_context
def validate(ctx, test_year, gender, schools):
    """Validate model against actual results (train on years before test-year)."""
    cfg = ctx.obj['config']
    
    train_years = [y for y in cfg.data['historical_years'] if y < test_year]
    
    if len(train_years) < 2:
        click.echo(f"Error: Need at least 2 years before {test_year} for training")
        return
    
    click.echo(f"\n{'='*60}")
    click.echo(f"Validating against {test_year}")
    click.echo(f"Training on: {train_years}")
    click.echo('='*60)
    
    gender_filter = None if gender == 'all' else gender
    school_filter = schools.split(',') if schools else None
    prev_year = test_year - 1
    
    # Load data
    df_all = build_dataset(train_years + [test_year], cfg.data['files_dir'])
    df_train = df_all[df_all['Year'].isin(train_years)].copy()
    df_test = df_all[df_all['Year'] == test_year].copy()
    
    if school_filter:
        mask = df_test['School'].str.contains('|'.join(school_filter), case=False, na=False)
        df_test = df_test[mask]
    
    # Filter to schools in training data
    train_schools = set(df_train['School'].unique())
    df_test = df_test[df_test['School'].isin(train_schools)].copy()
    
    click.echo(f"\nTraining samples: {len(df_train)}")
    click.echo(f"Test samples: {len(df_test)}\n")
    
    all_results = {}
    genders = get_gender_list(gender_filter)
    
    for round_num in [1, 2]:
        for g in genders:
            target_col = f'R{round_num}_Min_{g}'
            if target_col not in df_train.columns:
                continue
            
            click.echo(f"Training model for R{round_num} {g}...")
            
            # Train model
            model, le_school, le_profile, feature_cols, school_stats = train_model(
                df_train, target_col, round_num, cfg.model
            )
            
            if model is None:
                continue
            
            # Prepare test data
            X, df_prep, prev_scores_map, mask = prepare_prediction_data(
                df_test, df_train, target_col, prev_year,
                le_school, le_profile, school_stats, feature_cols
            )
            
            # Filter to valid test samples
            y_test = df_prep[target_col].fillna(0)
            valid_mask = y_test > 0
            
            # Generate predictions
            pred_results = generate_predictions(
                model, X, df_prep, prev_scores_map, school_stats, valid_mask
            )
            
            if not pred_results:
                continue
            
            # Compute metrics
            y_actual = np.array([df_test.loc[df_test['School'] == r.school].loc[
                df_test['Profile'] == r.profile, target_col].values[0] for r in pred_results])
            y_pred = np.array([r.predicted for r in pred_results])
            
            existing_mask = np.array([not r.is_new for r in pred_results])
            reliable_mask = np.array([r.reliable for r in pred_results])
            
            metrics_existing = compute_metrics(y_actual, y_pred, existing_mask)
            metrics_reliable = compute_metrics(y_actual, y_pred, reliable_mask)
            
            click.echo(f"  MAE: {metrics_existing['mae']:.2f} points (existing profiles)")
            click.echo(f"  MAE: {metrics_reliable['mae']:.2f} points (reliable only)")
            click.echo(f"  New profiles: {sum(r.is_new for r in pred_results)}/{len(pred_results)}")
            click.echo(f"  Reliable: {metrics_reliable['count']}/{len(pred_results)}\n")
            
            # Store detailed results (same format as predict)
            for r, actual in zip(pred_results, y_actual):
                key = (r.school, r.profile)
                if key not in all_results:
                    all_results[key] = {'School': r.school, 'Profile': r.profile}
                
                all_results[key][f'R{round_num}_{g}_Actual'] = actual
                all_results[key][f'R{round_num}_{g}_Predicted'] = round(r.predicted, 2)
                all_results[key][f'R{round_num}_{g}_Error'] = round(actual - r.predicted, 2)
                all_results[key][f'R{round_num}_{g}_Abs_Error'] = round(abs(actual - r.predicted), 2)
                all_results[key][f'R{round_num}_{g}_Volatility'] = round(r.volatility, 1)
                all_results[key][f'R{round_num}_{g}_Years_Data'] = r.n_years
                all_results[key][f'R{round_num}_{g}_Reliable'] = r.reliable
    
    if not all_results:
        click.echo("No results generated")
        return
    
    # Save results
    results_df = pd.DataFrame(list(all_results.values()))
    
    # Sort by R1 error
    sort_col = [c for c in results_df.columns if 'R1' in c and 'Abs_Error' in c]
    if sort_col:
        results_df = results_df.sort_values(sort_col[0], ascending=False)
    
    output_dir = Path(cfg.data['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    gender_suffix = gender_filter if gender_filter else "all"
    output_file = output_dir / f"Validation_{test_year}_{gender_suffix}.xlsx"
    results_df.to_excel(output_file, index=False)
    
    # Show worst predictions
    click.echo("=== Top 10 Worst Predictions (R1) ===")
    target = genders[0] if len(genders) == 1 else 'Female'
    display_cols = ['School', 'Profile', f'R1_{target}_Actual', f'R1_{target}_Predicted', f'R1_{target}_Error']
    display_cols = [c for c in display_cols if c in results_df.columns]
    abs_err_col = f'R1_{target}_Abs_Error'
    if abs_err_col in results_df.columns:
        worst = results_df.nlargest(10, abs_err_col)[display_cols]
        click.echo(worst.to_string(index=False))
    
    click.echo(f"\n✓ Detailed results saved to {output_file}")


@cli.command()
@click.option('--years', help='Comma-separated years to analyze')
@click.option('--gender', type=click.Choice(['male', 'female', 'all']))
@click.pass_context
def analyze(ctx, years, gender):
    """Analyze historical rankings data."""
    cfg = ctx.obj['config']
    
    from nvo.data.loaders import load_rankings
    
    years_list = [int(y) for y in years.split(',')] if years else cfg.data['historical_years'][-2:]
    gender_filter = None if gender == 'all' else (gender or cfg.filters.get('gender'))
    
    logger.info(f"Analyzing years: {years_list}")
    
    for year in years_list:
        df = load_rankings(year, cfg.data['files_dir'])
        if df is None:
            continue
        
        click.echo(f"\n=== Year {year} ===")
        click.echo(f"Total records: {len(df)}")
        click.echo(f"Schools: {df['School'].nunique()}")
        
        if gender_filter:
            target = 'Female' if gender_filter.lower().startswith('f') else 'Male'
            col = f'R1_Min_{target}'
            if col in df.columns:
                valid = df[df[col] > 0]
                click.echo(f"Valid {target} R1 scores: {len(valid)}")
                click.echo(f"Mean: {valid[col].mean():.1f}")
                click.echo(f"Median: {valid[col].median():.1f}")


if __name__ == '__main__':
    cli()
