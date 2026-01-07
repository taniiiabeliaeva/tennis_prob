"""
Gaussian Random Walk State-Space Model for Tennis Momentum
===========================================================

This script implements the Bayesian Multimodal State-Space Model that replaces
the static regression with a time-varying momentum model.

The model captures:
- Latent Momentum: A Gaussian Random Walk that evolves step-by-step
- Fixed Effects: Physical (fatigue) & Tactical (aggression, pressure) factors
"""

import pymc as pm
import numpy as np
import pickle
import arviz as az
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_data():
    """Load the prepared feature data."""
    input_path = PROCESSED_DIR / 'sinner_features_2024.pkl'
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data


def build_model(data):
    """
    Build the Multimodal State-Space Model (v2).
    
    The Mathematical Model:
    - alpha_t ~ GaussianRandomWalk(sigma_drift) : Latent momentum
    - logit(p_t) = alpha_t + beta_2nd * second_serve + beta_depth * depth + beta_rally * rally 
                   + beta_aggr * aggression + beta_pressure * pressure
    - y_t ~ Bernoulli(p_t)
    
    Features (v2):
    - second_serve: Binary - is this a 2nd serve?
    - depth_pressure: Binary - did opponent return deep?
    - prev_rally_length: Scalar - how long was the previous rally?
    - aggression: Binary - did server aim for the lines?
    - pressure: Binary - is it a break point/deuce?
    """
    y_obs = data['y_obs']
    second_serve = data['second_serve_obs']
    depth_pressure = data['depth_pressure_obs']
    prev_rally = data['prev_rally_obs']
    aggression = data['aggr_obs']
    pressure = data['pressure_obs']
    n_points = len(y_obs)
    
    print(f"Building model with {n_points} observations...")
    print(f"  Win rate: {y_obs.mean():.3f}")
    print(f"  Second serve rate: {second_serve.mean():.1%}")
    print(f"  Deep return rate: {depth_pressure.mean():.1%}")
    print(f"  Prev rally mean: {prev_rally.mean():.3f}")
    print(f"  Aggression rate: {aggression.mean():.1%}")
    print(f"  Pressure rate: {pressure.mean():.1%}")
    
    with pm.Model() as dynamic_model:
        # --- A. Latent Momentum (Non-Centered Parameterization) ---
        # This fixes the "funnel degeneracy" where the sampler gets stuck
        # trying to jointly estimate alpha and sigma.
        
        # Volatility: How fast does Sinner's form change?
        # Small sigma = Stable form; Large sigma = Erratic.
        sigma_drift = pm.HalfNormal("sigma_drift", sigma=0.1)
        
        # NON-CENTERED PARAMETERIZATION:
        # 1. Sample "raw" steps from a standard Normal(0, 1)
        #    These are easy for the sampler to explore because they have fixed scale.
        raw_steps = pm.Normal("raw_steps", mu=0, sigma=1, shape=n_points)
        
        # 2. Scale them by sigma_drift and accumulate
        #    This creates the Random Walk deterministically
        import pytensor.tensor as pt
        random_walk = pm.Deterministic("random_walk", pt.cumsum(raw_steps * sigma_drift))
        
        # --- B. Fixed Effects (Context-Dependent Modifiers) ---
        
        # NEW: Second Serve Penalty (Expect NEGATIVE)
        # Missing the first serve immediately drops win probability
        beta_second_serve = pm.Normal("beta_second_serve", mu=0, sigma=1)
        
        # NEW: Depth Pressure (Expect NEGATIVE)
        # Deep returns from Medvedev hurt Sinner
        beta_depth = pm.Normal("beta_depth", mu=0, sigma=1)
        
        # NEW: Previous Rally Length (Expect NEGATIVE)
        # Long rallies cause immediate fatigue for the next point
        beta_rally = pm.Normal("beta_rally", mu=0, sigma=1)
        
        # EXISTING: Aggression (Ambiguous - High Risk/Reward)
        beta_aggr = pm.Normal("beta_aggr", mu=0, sigma=1)
        
        # EXISTING: Pressure (Expect NEGATIVE for "choking")
        beta_pressure = pm.Normal("beta_pressure", mu=0, sigma=1)
        
        # --- C. Link Function ---
        # Combine the Time-Varying Momentum with the Contextual States
        logit_p = random_walk + \
                  (beta_second_serve * second_serve) + \
                  (beta_depth * depth_pressure) + \
                  (beta_rally * prev_rally) + \
                  (beta_aggr * aggression) + \
                  (beta_pressure * pressure)
        
        # --- D. Likelihood ---
        y_est = pm.Bernoulli("y_est", logit_p=logit_p, observed=y_obs)
    
    return dynamic_model


def run_inference(model, draws=1000, tune=1000, chains=4, target_accept=0.95):
    """Run MCMC inference using NUTS sampler."""
    print(f"\nStarting MCMC inference...")
    print(f"  Draws: {draws}")
    print(f"  Tune: {tune}")
    print(f"  Chains: {chains}")
    print(f"  Target accept: {target_accept}")
    
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True
        )
    
    return trace


def check_diagnostics(trace):
    """Check MCMC diagnostics."""
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    
    # Divergence check
    divergences = trace.sample_stats['diverging'].sum().values
    print(f"Divergences: {divergences}")
    if divergences > 0:
        print(f"  ⚠ Warning: {divergences} divergences detected. Consider increasing target_accept.")
    else:
        print("  ✓ No divergences")
    
    # Summary statistics
    print("\n" + "-" * 60)
    print("PARAMETER SUMMARY")
    print("-" * 60)
    summary = az.summary(
        trace, 
        var_names=["sigma_drift", "beta_second_serve", "beta_depth", 
                   "beta_rally", "beta_aggr", "beta_pressure"]
    )
    print(summary)
    
    # Check R-hat
    rhat_issues = summary[summary['r_hat'] > 1.01]
    if len(rhat_issues) > 0:
        print(f"\n  ⚠ Warning: R-hat > 1.01 for some parameters:")
        print(rhat_issues)
    else:
        print("\n  ✓ All R-hat values < 1.01 (good convergence)")
    
    # Check ESS
    ess_issues = summary[summary['ess_bulk'] < 400]
    if len(ess_issues) > 0:
        print(f"\n  ⚠ Warning: ESS < 400 for some parameters:")
        print(ess_issues)
    else:
        print("  ✓ All ESS values > 400 (good mixing)")
    
    return summary


def interpret_results(summary):
    """
    Interpret the model results using HDI-based credibility.
    
    A parameter is "statistically credible" if the 94% HDI excludes zero.
    """
    print("\n" + "=" * 60)
    print("INTERPRETATION (HDI-Based Credibility)")
    print("=" * 60)
    
    def check_credibility(param_name, description):
        """Check if HDI excludes zero and report credibility."""
        mean_val = summary.loc[param_name, 'mean']
        hdi_low = summary.loc[param_name, 'hdi_3%']
        hdi_high = summary.loc[param_name, 'hdi_97%']
        
        # Check if HDI excludes zero
        if hdi_low > 0:
            status = "✓✓ CREDIBLE POSITIVE"
            excludes_zero = True
        elif hdi_high < 0:
            status = "✓✓ CREDIBLE NEGATIVE"
            excludes_zero = True
        else:
            status = "⚠ INCONCLUSIVE"
            excludes_zero = False
        
        print(f"\n{param_name} ({description}):")
        print(f"   Mean: {mean_val:+.3f}, 94% HDI: [{hdi_low:+.3f}, {hdi_high:+.3f}]")
        print(f"   Status: {status}")
        
        return excludes_zero, mean_val
    
    # Check each parameter - NEW features first
    print("\n--- NEW FEATURES (v2) ---")
    _, _ = check_credibility('beta_second_serve', 'Second Serve Penalty')
    _, _ = check_credibility('beta_depth', 'Opponent Deep Returns')
    _, _ = check_credibility('beta_rally', 'Previous Rally Load')
    
    print("\n--- EXISTING FEATURES ---")
    _, _ = check_credibility('beta_aggr', 'Aggression')
    _, _ = check_credibility('beta_pressure', 'Pressure Points')
    
    # Sigma drift (momentum volatility)
    sigma_drift_mean = summary.loc['sigma_drift', 'mean']
    sigma_drift_rhat = summary.loc['sigma_drift', 'r_hat']
    sigma_drift_ess = summary.loc['sigma_drift', 'ess_bulk']
    
    print(f"\nsigma_drift (Momentum Volatility):")
    print(f"   Mean: {sigma_drift_mean:.3f}")
    print(f"   R-hat: {sigma_drift_rhat:.2f}, ESS: {sigma_drift_ess:.0f}")
    
    if sigma_drift_rhat > 1.05 or sigma_drift_ess < 100:
        print(f"   ⚠ CONVERGENCE WARNING: R-hat should be ~1.00, ESS should be >400")
        print(f"   ➤ The non-centered parameterization should fix this. Please re-run inference.")
    else:
        print(f"   ✓ Good convergence")
        if sigma_drift_mean < 0.05:
            print(f"   ➤ Momentum is STABLE")
        elif sigma_drift_mean > 0.15:
            print(f"   ➤ Momentum is VOLATILE")
        else:
            print(f"   ➤ Momentum is MODERATE")


def save_results(trace, summary):
    """Save the trace and summary to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save trace
    trace_path = RESULTS_DIR / 'trace_multimodal.nc'
    trace.to_netcdf(trace_path)
    print(f"\nTrace saved to: {trace_path}")
    
    # Save summary
    summary_path = RESULTS_DIR / 'summary_multimodal.csv'
    summary.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")


def main():
    """Main function to run the Bayesian inference pipeline."""
    print("=" * 60)
    print("MULTIMODAL STATE-SPACE MODEL - BAYESIAN INFERENCE")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = load_data()
    print(f"   Loaded {len(data['y_obs'])} observations")
    
    # Build model
    print("\n2. Building model...")
    model = build_model(data)
    
    # Run inference
    print("\n3. Running inference...")
    trace = run_inference(model)
    
    # Check diagnostics
    summary = check_diagnostics(trace)
    
    # Interpret results
    interpret_results(summary)
    
    # Save results
    print("\n4. Saving results...")
    save_results(trace, summary)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    
    return trace, summary


if __name__ == "__main__":
    trace, summary = main()
