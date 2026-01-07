"""
Comprehensive Match Visualization
=================================

Generates publication-quality plots for the Alcaraz 2024 Final analysis:
1. Mental Worm (Momentum + Pressure)
2. Rolling Win Rate vs Model Prediction
3. Feature Impact Comparison
4. Summary Dashboard
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import arviz as az
from pathlib import Path
from scipy.special import expit  # logistic function

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'momentum': '#d62728',
    'clutch': '#2ca02c',
    'choke': '#1f1f1f',
    'win_rate': '#1f77b4',
    'model': '#ff7f0e',
    'pressure': '#9467bd'
}


def load_data_and_trace():
    """Load target data and Stage 2 trace."""
    target_path = PROCESSED_DIR / 'alcaraz_target_2024_final.pkl'
    with open(target_path, 'rb') as f:
        target_data = pickle.load(f)
    
    trace_path = RESULTS_DIR / 'trace_stage2.nc'
    trace = az.from_netcdf(trace_path)
    
    return target_data, trace


def calculate_predicted_probability(target_data, trace):
    """Calculate predicted win probability from the model."""
    # Get posterior samples for all parameters
    random_walk = trace.posterior["random_walk"].mean(dim=["chain", "draw"]).values
    beta_second = trace.posterior["beta_second_serve"].mean(dim=["chain", "draw"]).values
    beta_depth = trace.posterior["beta_depth"].mean(dim=["chain", "draw"]).values
    beta_rally = trace.posterior["beta_rally"].mean(dim=["chain", "draw"]).values
    beta_aggr = trace.posterior["beta_aggr"].mean(dim=["chain", "draw"]).values
    beta_pressure = trace.posterior["beta_pressure"].mean(dim=["chain", "draw"]).values
    
    # Calculate logit
    logit_p = random_walk + \
              beta_second * target_data['second_serve_obs'] + \
              beta_depth * target_data['depth_pressure_obs'] + \
              beta_rally * target_data['prev_rally_obs'] + \
              beta_aggr * target_data['aggr_obs'] + \
              beta_pressure * target_data['pressure_obs']
    
    # Convert to probability
    prob = expit(logit_p)
    return prob


def rolling_win_rate(y_obs, window=10):
    """Calculate rolling win rate with specified window."""
    kernel = np.ones(window) / window
    rolling = np.convolve(y_obs, kernel, mode='valid')
    # Pad to match original length
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.concatenate([
        np.full(pad_left, np.nan),
        rolling,
        np.full(pad_right, np.nan)
    ])
    return padded[:len(y_obs)]


def plot_comprehensive_dashboard(target_data, trace, save_path=None):
    """
    Create a comprehensive 4-panel dashboard:
    1. Momentum Trajectory (Mental Worm)
    2. Win Rate: Observed vs Predicted
    3. Effect Magnitudes (Forest Plot)
    4. Pressure Impact (Bar Chart)
    """
    # Prepare data
    n_points = len(target_data['y_obs'])
    y_obs = target_data['y_obs']
    pressure = target_data['pressure_obs']
    
    # Get momentum
    random_walk = trace.posterior["random_walk"]
    momentum_mean = random_walk.mean(dim=["chain", "draw"]).values
    hdi = az.hdi(random_walk, hdi_prob=0.94)["random_walk"].values
    
    # Get predicted probability
    pred_prob = calculate_predicted_probability(target_data, trace)
    
    # Calculate rolling win rates
    rolling_10 = rolling_win_rate(y_obs, window=10)
    rolling_20 = rolling_win_rate(y_obs, window=20)
    
    # Get parameter summary
    summary = az.summary(trace, var_names=[
        "beta_second_serve", "beta_depth", "beta_rally", "beta_aggr", "beta_pressure"
    ])
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    # ============================================
    # Panel 1: Momentum Trajectory (Mental Worm)
    # ============================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Momentum line
    ax1.plot(momentum_mean, color=COLORS['momentum'], lw=2, label='Momentum')
    ax1.fill_between(range(n_points), hdi[:, 0], hdi[:, 1], 
                     color=COLORS['momentum'], alpha=0.15, label='94% HDI')
    
    # Pressure points
    pressure_idx = np.where(pressure == 1)[0]
    pressure_wins = y_obs[pressure_idx] == 1
    
    ax1.scatter(pressure_idx[pressure_wins], momentum_mean[pressure_idx[pressure_wins]],
                s=120, c=COLORS['clutch'], edgecolors='white', linewidths=2,
                zorder=5, label='Clutch')
    ax1.scatter(pressure_idx[~pressure_wins], momentum_mean[pressure_idx[~pressure_wins]],
                s=120, c=COLORS['choke'], marker='X', linewidths=2,
                zorder=5, label='Choke')
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('A. Latent Momentum (Random Walk)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Service Point')
    ax1.set_ylabel('Log-Odds Advantage')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(-2, n_points + 2)
    
    # ============================================
    # Panel 2: Win Rate - Observed vs Predicted
    # ============================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Rolling win rates
    ax2.plot(rolling_10, color=COLORS['win_rate'], lw=2, alpha=0.7, 
             label='Rolling Win Rate (10pt)')
    ax2.plot(rolling_20, color=COLORS['win_rate'], lw=2.5, 
             label='Rolling Win Rate (20pt)')
    
    # Model prediction
    ax2.plot(pred_prob, color=COLORS['model'], lw=2, linestyle='--', 
             label='Model Predicted P(win)')
    
    # Point outcomes as scatter
    ax2.scatter(np.where(y_obs == 1)[0], np.ones(np.sum(y_obs == 1)) * 1.02,
                color=COLORS['clutch'], alpha=0.4, s=15, marker='|')
    ax2.scatter(np.where(y_obs == 0)[0], np.zeros(np.sum(y_obs == 0)) - 0.02,
                color=COLORS['choke'], alpha=0.4, s=15, marker='|')
    
    # Overall win rate line
    ax2.axhline(y_obs.mean(), color='gray', linestyle=':', alpha=0.7, 
                label=f'Overall: {y_obs.mean():.1%}')
    
    ax2.set_title('B. Win Rate Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Service Point')
    ax2.set_ylabel('Win Probability')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_ylim(-0.1, 1.15)
    ax2.set_xlim(-2, n_points + 2)
    
    # ============================================
    # Panel 3: Effect Magnitudes (Forest Plot)
    # ============================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    params = ['beta_pressure', 'beta_second_serve', 'beta_depth', 'beta_aggr', 'beta_rally']
    labels = ['Pressure\n(Big Points)', 'Second Serve\nPenalty', 'Deep Returns\n(Opponent)', 
              'Aggression\n(Line-Hitting)', 'Rally Length\n(Fatigue)']
    
    y_pos = np.arange(len(params))
    means = [summary.loc[p, 'mean'] for p in params]
    hdi_lows = [summary.loc[p, 'hdi_3%'] for p in params]
    hdi_highs = [summary.loc[p, 'hdi_97%'] for p in params]
    
    # Error bars
    errors = [[m - l for m, l in zip(means, hdi_lows)],
              [h - m for m, h in zip(means, hdi_highs)]]
    
    # Colors based on whether HDI excludes zero
    colors = []
    for i, p in enumerate(params):
        if hdi_highs[i] < 0:
            colors.append('#d62728')  # Credible negative
        elif hdi_lows[i] > 0:
            colors.append('#2ca02c')  # Credible positive
        else:
            colors.append('#7f7f7f')  # Inconclusive
    
    ax3.barh(y_pos, means, xerr=errors, color=colors, alpha=0.7, 
             edgecolor='black', linewidth=1, capsize=4)
    ax3.axvline(0, color='black', linestyle='-', lw=1.5)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=10)
    ax3.set_xlabel('Effect Size (Log-Odds)', fontsize=11)
    ax3.set_title('C. Effect Magnitudes with 94% HDI', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    # Add credibility markers
    for i, p in enumerate(params):
        if hdi_highs[i] < 0 or hdi_lows[i] > 0:
            ax3.annotate('✓', xy=(means[i] + 0.1, i), fontsize=14, 
                        color='darkgreen' if hdi_lows[i] > 0 else 'darkred',
                        fontweight='bold', va='center')
    
    # ============================================
    # Panel 4: Pressure Context Analysis
    # ============================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate win rates by context
    normal_pts = pressure == 0
    pressure_pts = pressure == 1
    
    contexts = ['Normal\nPoints', 'Pressure\nPoints', 'Overall']
    win_rates = [
        y_obs[normal_pts].mean(),
        y_obs[pressure_pts].mean(),
        y_obs.mean()
    ]
    counts = [
        f'n={np.sum(normal_pts)}',
        f'n={np.sum(pressure_pts)}',
        f'n={len(y_obs)}'
    ]
    
    bars = ax4.bar(contexts, win_rates, color=[COLORS['win_rate'], COLORS['pressure'], 'gray'],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add win rate labels
    for bar, rate, count in zip(bars, win_rates, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=12, fontweight='bold')
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                count, ha='center', fontsize=10, color='white', fontweight='bold')
    
    # Add pressure drop arrow
    ax4.annotate('', xy=(1, win_rates[1]), xytext=(0, win_rates[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(0.5, (win_rates[0] + win_rates[1])/2 + 0.08, 
            f'−{(win_rates[0] - win_rates[1])*100:.0f}pp',
            ha='center', fontsize=11, color='red', fontweight='bold')
    
    ax4.set_ylabel('Win Rate', fontsize=11)
    ax4.set_title('D. Pressure Impact Analysis', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    
    # Overall title
    fig.suptitle('Carlos Alcaraz: 2024 Wimbledon Final vs Djokovic\n'
                 'Hierarchical State-Space Model Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Dashboard saved to: {save_path}")
    
    plt.show()
    return fig


def main():
    """Generate comprehensive visualization dashboard."""
    print("=" * 60)
    print("COMPREHENSIVE MATCH VISUALIZATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data and trace...")
    target_data, trace = load_data_and_trace()
    
    # Generate dashboard
    print("Generating 4-panel dashboard...")
    save_path = RESULTS_DIR / 'alcaraz_2024_final_dashboard.png'
    plot_comprehensive_dashboard(target_data, trace, save_path)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
