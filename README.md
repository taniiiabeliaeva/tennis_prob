# From Static Skill to Dynamic Momentum: A Hierarchical Bayesian Analysis of Tennis

## Project Overview

This project applies **Probabilistic Programming** to analyze the evolution of player performance during tennis matches. We use Bayesian inference to model latent momentum dynamics and quantify the effects of **pressure situations** and **aggressive serve placement** on point outcomes.

Unlike static post-match statistics, our models update beliefs point-by-point, revealing how player consistency shifts dynamically throughout a match.

## Research Questions

- **RQ1:** Does performance systematically differ under pressure (break/set/match points)?
- **RQ2:** Does aggressive serve placement provide a measurable advantage over conservative placement?

---

## Project Structure

```text
tennis_prob/
├── data/
│   ├── raw/                              # Original CSVs from Jeff Sackmann dataset
│   │   ├── 2016-wimbledon-{matches,points}.csv
│   │   ├── ...
│   │   └── 2024-wimbledon-{matches,points}.csv
│   │
│   └── processed/                        # Cleaned and engineered data
│       ├── 2023-wimbledon-points-corrected.csv
│       ├── 2024-wimbledon-points-corrected.csv
│       ├── match_features_enhanced.pkl   # Single-match features (pressure, aggression)
│       ├── hierarchical_features.pkl     # Multi-year features for hierarchical model
│       └── priors_2023.pkl               # Extracted Beta priors from historical data
│
├── notebooks_src/
│   ├── 01_data_prior_extraction.ipynb    # Data cleaning & historical prior extraction
│   ├── 02_match_data_engineering.ipynb   # Feature engineering (pressure, aggression)
│   ├── 03_bayesian_inference.ipynb       # PyMC models and MCMC sampling
│   ├── 04_posterior_analysis.ipynb       # Posterior analysis and visualizations
│   └── match_probability.py              # Helper functions for probability calculations
│
├── results/                              # Model outputs and visualizations
│   ├── summary_hierarchical.csv          # Hierarchical model summary statistics
│   ├── player_effects_hierarchical.csv   # Player-specific effect estimates
│   └── *.png                             # Visualization outputs
│
├── requirements.txt
└── README.md
```

---

## Methodology

### Two-Stage Bayesian Framework

We develop a two-stage modeling approach to capture both within-match dynamics and population-level effects.

### Stage 1: GRW State-Space Model (Single-Match)

For the case study (Sinner vs. Medvedev, 2024 Wimbledon QF, 326 points), we model latent momentum using a **Gaussian Random Walk (GRW)**:

$$\epsilon_t \sim \mathcal{N}(0, 1)$$

$$m_t = \sigma_{drift} \cdot \sum_{i=1}^{t} \epsilon_i \quad \text{(Latent Momentum)}$$

$$\text{logit}(\theta_t) = m_t + \beta_{pressure} \cdot P_t + \beta_{aggr} \cdot A_t$$

$$y_t \sim \text{Bernoulli}(\theta_t)$$

We use **non-centered parameterization** to avoid funnel geometries that impede MCMC sampling.

### Stage 2: Hierarchical Bayesian Model (Population)

To leverage the full dataset (~30,000 serve points from 2016-2024), we treat player effects as drawn from a population distribution:

$$\alpha_j \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha) \quad \text{(Player baseline)}$$

$$\beta_{pressure}^{(j)} \sim \mathcal{N}(\mu_{pressure}, \sigma_{pressure}) \quad \text{(Player pressure effect)}$$

$$\beta_{aggr}^{(j)} \sim \mathcal{N}(\mu_{aggr}, \sigma_{aggr}) \quad \text{(Player aggression effect)}$$

This **partial pooling** approach allows estimation of player-specific heterogeneity while shrinking estimates toward the group mean.

### Feature Engineering

Two binary covariates are constructed from raw tracking data:

- **Pressure ($P_t$):** Coded as 1 if the point is a break, set, or match point; 0 otherwise
- **Aggression ($A_t$):** Coded as 1 for high-risk serve zones (wide or center-T) and 0 for conservative (body/middle) zones

---

## Key Findings

### The Pressure Effect ($\mu_{pressure} = -0.531$)

Servers perform significantly **worse** under pressure:
- **Baseline:** A standard server has a win probability of ~57.2%
- **Under Pressure:** Probability drops to ~44.0%
- This represents a **13.2 percentage point decline**

The low standard deviation suggests this effect is consistent across the ATP tour—even elite performers struggle during high-stakes moments.

### The Aggression Effect ($\mu_{aggr} = 0.771$)

Aggressive serve placement yields a measurable advantage:
- **Conservative:** Win probability ~57.2%
- **Aggressive:** Win probability rises to ~74.2%
- This represents a **17 percentage point gain**

The moderate heterogeneity indicates this benefit is not uniform—it favors players with the physical capacity to execute these serves consistently.

### Player-Specific Effects

Partial pooling reveals individual deviations from population means:
- **Best Aggressive Servers:** John Isner, Tomas Berdych, Kevin Anderson
- **Most Pressure-Resilient:** Carlos Alcaraz, Jannik Sinner, Kei Nishikori

---

## Usage Guide

### Prerequisites

```bash
pip install -r requirements.txt
```

### Execute Notebooks in Order

**1. `01_data_prior_extraction.ipynb`**
- **Input:** Raw CSV files in `data/raw/`
- **Action:** Reconstructs missing serve statistics using ServeNumber logic; calculates Beta priors from 2023 data
- **Output:** `priors_2023.pkl`, corrected point CSVs

**2. `02_match_data_engineering.ipynb`**
- **Input:** Corrected 2024 CSV, priors pickle
- **Action:** Filters target match (Sinner vs Medvedev); engineers pressure and aggression features; prepares hierarchical dataset
- **Output:** `match_features_enhanced.pkl`, `hierarchical_features.pkl`

**3. `03_bayesian_inference.ipynb`**
- **Input:** Feature pickle files
- **Action:** Builds and runs both GRW State-Space and Hierarchical Bayesian models using PyMC/NUTS
- **Output:** Inference traces (NetCDF), summary CSVs, model comparison

**4. `04_posterior_analysis.ipynb`**
- **Input:** Inference traces, feature data
- **Action:** Analyzes posteriors; creates visualizations for momentum trajectories, population effects, and player-specific coefficients
- **Output:** PNG visualizations, interpretation pickles

---

## Inference Details

- **Algorithm:** NUTS (No-U-Turn Sampler) via PyMC v5
- **Configuration:** 4 chains, 1000 tuning + 1000/2000 draws
- **Convergence:** All parameters achieved $\hat{R} < 1.01$
- **Diagnostics:** Zero divergent transitions; ESS > 3000 for population parameters

---

## Data Source

This project uses the **Grand Slam Point-by-Point Data** maintained by Jeff Sackmann.

- **Repository:** [github.com/JeffSackmann/tennis_slam_pointbypoint](https://github.com/JeffSackmann/tennis_slam_pointbypoint)
- **Scope:** Wimbledon Championships 2016–2024, filtered to Round of 16 and beyond (~30,000 serve points, 100+ players)

---

## Results Visualizations

| Output | Description |
|--------|-------------|
| `hierarchical_population_effects_analysis.png` | Posterior distributions for pressure and aggression effects |
| `hierarchical_player_effects.png` | Top players by pressure resilience and aggression benefit |
| `match_win_probability_corrected.png` | Point-by-point win probability for case study |
| `momentum_trajectory_grw.png` | Latent momentum evolution during the match |

---

## Authors

- Tatiana Beliaeva (12432964)
- Ege Aydin (12432147)

## License

This project is for academic purposes (TU Wien 2025WS).
