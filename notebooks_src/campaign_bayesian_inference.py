"""
Campaign Bayesian Inference - Transfer Learning
================================================

Two-stage inference for transfer learning:
- Stage 1: Learn physics (betas) from training data (simple logistic regression)
- Stage 2: Apply learned priors to target match with Random Walk momentum

Usage:
    python campaign_bayesian_inference.py --stage=1  # Learn from training data
    python campaign_bayesian_inference.py --stage=2  # Apply to target match
    python campaign_bayesian_inference.py            # Run both stages
"""

import pymc as pm
import numpy as np
import pickle
import json
import arviz as az
import pytensor.tensor as pt
from pathlib import Path
import argparse

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_training_data():
    """Load the training dataset (13 matches)."""
    path = PROCESSED_DIR / 'alcaraz_training_2023_2024.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_target_data():
    """Load the target dataset (2024 Final)."""
    path = PROCESSED_DIR / 'alcaraz_target_2024_final.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_stage1_model(data):
    """
    Stage 1: Simple Logistic Regression (No Random Walk)
    
    Learn physics parameters from pooled training data.
    No momentum modeling here - just the fixed effects.
    """
    y_obs = data['y_obs']
    second_serve = data['second_serve_obs']
    depth_pressure = data['depth_pressure_obs']
    prev_rally = data['prev_rally_obs']
    aggression = data['aggr_obs']
    pressure = data['pressure_obs']
    n_points = len(y_obs)
    
    print(f"Stage 1: Building logistic regression with {n_points} observations...")
    print(f"  Win rate: {y_obs.mean():.1%}")
    
    with pm.Model() as stage1_model:
        # Intercept (baseline win probability)
        intercept = pm.Normal("intercept", mu=0.5, sigma=1)
        
        # Physics parameters (what we want to learn)
        beta_second_serve = pm.Normal("beta_second_serve", mu=0, sigma=1)
        beta_depth = pm.Normal("beta_depth", mu=0, sigma=1)
        beta_rally = pm.Normal("beta_rally", mu=0, sigma=1)
        beta_aggr = pm.Normal("beta_aggr", mu=0, sigma=1)
        beta_pressure = pm.Normal("beta_pressure", mu=0, sigma=1)
        
        # Link function (no random walk noise)
        logit_p = intercept + \
                  (beta_second_serve * second_serve) + \
                  (beta_depth * depth_pressure) + \
                  (beta_rally * prev_rally) + \
                  (beta_aggr * aggression) + \
                  (beta_pressure * pressure)
        
        # Likelihood
        y_est = pm.Bernoulli("y_est", logit_p=logit_p, observed=y_obs)
    
    return stage1_model


def build_stage2_model(data, learned_priors):
    """
    Stage 2: Random Walk with Informed Priors
    
    Apply learned physics to target match with full momentum modeling.
    """
    y_obs = data['y_obs']
    second_serve = data['second_serve_obs']
    depth_pressure = data['depth_pressure_obs']
    prev_rally = data['prev_rally_obs']
    aggression = data['aggr_obs']
    pressure = data['pressure_obs']
    n_points = len(y_obs)
    
    print(f"Stage 2: Building Random Walk model with {n_points} observations...")
    print(f"  Win rate: {y_obs.mean():.1%}")
    print(f"  Using learned priors from Stage 1")
    
    with pm.Model() as stage2_model:
        # --- Latent Momentum (Non-Centered Parameterization) ---
        sigma_drift = pm.HalfNormal("sigma_drift", sigma=0.1)
        raw_steps = pm.Normal("raw_steps", mu=0, sigma=1, shape=n_points)
        random_walk = pm.Deterministic("random_walk", pt.cumsum(raw_steps * sigma_drift))
        
        # --- Physics with INFORMED priors (from Stage 1) ---
        beta_second_serve = pm.Normal(
            "beta_second_serve", 
            mu=learned_priors['beta_second_serve']['mean'],
            sigma=learned_priors['beta_second_serve']['sd'] * 2  # Slightly wider
        )
        beta_depth = pm.Normal(
            "beta_depth", 
            mu=learned_priors['beta_depth']['mean'],
            sigma=learned_priors['beta_depth']['sd'] * 2
        )
        beta_rally = pm.Normal(
            "beta_rally", 
            mu=learned_priors['beta_rally']['mean'],
            sigma=learned_priors['beta_rally']['sd'] * 2
        )
        beta_aggr = pm.Normal(
            "beta_aggr", 
            mu=learned_priors['beta_aggr']['mean'],
            sigma=learned_priors['beta_aggr']['sd'] * 2
        )
        beta_pressure = pm.Normal(
            "beta_pressure", 
            mu=learned_priors['beta_pressure']['mean'],
            sigma=learned_priors['beta_pressure']['sd'] * 2
        )
        
        # --- Link function with Random Walk ---
        logit_p = random_walk + \
                  (beta_second_serve * second_serve) + \
                  (beta_depth * depth_pressure) + \
                  (beta_rally * prev_rally) + \
                  (beta_aggr * aggression) + \
                  (beta_pressure * pressure)
        
        # Likelihood
        y_est = pm.Bernoulli("y_est", logit_p=logit_p, observed=y_obs)
    
    return stage2_model


def run_inference(model, draws=1000, tune=1000, chains=4, target_accept=0.95):
    """Run MCMC inference."""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune, 
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True
        )
    return trace


def extract_posteriors(trace, var_names):
    """Extract posterior means and SDs for transfer learning."""
    priors = {}
    for var in var_names:
        samples = trace.posterior[var].values.flatten()
        priors[var] = {
            'mean': float(np.mean(samples)),
            'sd': float(np.std(samples)),
            'hdi_low': float(np.percentile(samples, 3)),
            'hdi_high': float(np.percentile(samples, 97))
        }
    return priors


def print_summary(trace, var_names, stage_name):
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"{stage_name} - RESULTS")
    print("=" * 60)
    
    summary = az.summary(trace, var_names=var_names)
    print(summary)
    
    # Check convergence
    rhat_ok = all(summary['r_hat'] < 1.01)
    ess_ok = all(summary['ess_bulk'] > 400)
    
    print()
    if rhat_ok and ess_ok:
        print("✓ Good convergence (R-hat < 1.01, ESS > 400)")
    else:
        print("⚠ Convergence issues detected")
    
    return summary


def run_stage1():
    """Execute Stage 1: Learn physics from training data."""
    print("=" * 60)
    print("STAGE 1: LEARNING PHYSICS FROM TRAINING DATA")
    print("=" * 60)
    
    # Load training data
    data = load_training_data()
    
    # Build and run model
    model = build_stage1_model(data)
    trace = run_inference(model)
    
    # Print summary
    var_names = ["intercept", "beta_second_serve", "beta_depth", 
                 "beta_rally", "beta_aggr", "beta_pressure"]
    summary = print_summary(trace, var_names, "STAGE 1")
    
    # Extract posteriors for Stage 2
    priors = extract_posteriors(trace, var_names[1:])  # Exclude intercept
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    priors_path = RESULTS_DIR / 'learned_priors.json'
    with open(priors_path, 'w') as f:
        json.dump(priors, f, indent=2)
    print(f"\nLearned priors saved to: {priors_path}")
    
    trace_path = RESULTS_DIR / 'trace_stage1.nc'
    trace.to_netcdf(trace_path)
    print(f"Trace saved to: {trace_path}")
    
    # Show learned priors
    print("\n" + "-" * 60)
    print("LEARNED PRIORS (for Stage 2)")
    print("-" * 60)
    for param, values in priors.items():
        print(f"  {param}: N({values['mean']:+.3f}, {values['sd']:.3f})")
    
    return trace, priors


def run_stage2(learned_priors=None):
    """Execute Stage 2: Apply to target match with Random Walk."""
    print("\n" + "=" * 60)
    print("STAGE 2: APPLYING TO TARGET MATCH WITH RANDOM WALK")
    print("=" * 60)
    
    # Load learned priors if not provided
    if learned_priors is None:
        priors_path = RESULTS_DIR / 'learned_priors.json'
        with open(priors_path, 'r') as f:
            learned_priors = json.load(f)
        print(f"Loaded priors from: {priors_path}")
    
    # Load target data
    data = load_target_data()
    
    # Build and run model
    model = build_stage2_model(data, learned_priors)
    trace = run_inference(model)
    
    # Print summary
    var_names = ["sigma_drift", "beta_second_serve", "beta_depth", 
                 "beta_rally", "beta_aggr", "beta_pressure"]
    summary = print_summary(trace, var_names, "STAGE 2")
    
    # Save results
    trace_path = RESULTS_DIR / 'trace_stage2.nc'
    trace.to_netcdf(trace_path)
    print(f"\nTrace saved to: {trace_path}")
    
    summary_path = RESULTS_DIR / 'summary_campaign.csv'
    summary.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION (with Transfer Learning)")
    print("=" * 60)
    
    for var in var_names[1:]:  # Skip sigma_drift
        mean_val = summary.loc[var, 'mean']
        hdi_low = summary.loc[var, 'hdi_3%']
        hdi_high = summary.loc[var, 'hdi_97%']
        
        if hdi_low > 0:
            status = "✓✓ CREDIBLE POSITIVE"
        elif hdi_high < 0:
            status = "✓✓ CREDIBLE NEGATIVE"
        else:
            status = "⚠ INCONCLUSIVE"
        
        print(f"\n{var}:")
        print(f"   Mean: {mean_val:+.3f}, 94% HDI: [{hdi_low:+.3f}, {hdi_high:+.3f}]")
        print(f"   Status: {status}")
    
    return trace


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Campaign Bayesian Inference')
    parser.add_argument('--stage', type=int, choices=[1, 2], 
                        help='Run specific stage (1 or 2). Default: both')
    args = parser.parse_args()
    
    if args.stage == 1:
        run_stage1()
    elif args.stage == 2:
        run_stage2()
    else:
        # Run both stages
        trace1, priors = run_stage1()
        trace2 = run_stage2(priors)
        
        print("\n" + "=" * 60)
        print("TRANSFER LEARNING COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
