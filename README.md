# Bayesian Momentum Analysis: Sinner vs. Medvedev (Wimbledon 2024)

## Project Overview
This project applies **Probabilistic Programming** to analyze the evolution of player performance during a high-stakes tennis match. Specifically, we use Bayesian inference to model the latent "serve momentum" (First Serve % and Ace Rate) of Jannik Sinner and Daniil Medvedev during their 2024 Wimbledon encounter.

Unlike static post-match statistics, our model updates its beliefs point-by-point, revealing how player consistency shifts dynamically throughout the game.

## Research Question
*Does updating Bayesian posteriors with in-game technical statistics reduce uncertainty in point-level win probability faster than a baseline average?*

---

##  Project Structure

```text
project_root/
├── data/
│   ├── raw/                        # Original CSVs from Jeff Sackmann dataset
│   │ 
│   └── processed/              
│       ├── 2023-wimbledon-points-corrected.csv  # Fixed ServeNumber logic
│       ├── match_1501_features.pkl              # Encoded observations for PyMC
│       └── priors_2023.pkl                      # Extracted Beta priors
│
├── notebooks_src/                 
│   ├── 01_data_prior_extraction.ipynb   # Data cleaning & historical prior extraction
│   ├── 02_match_data_engineering.ipynb  # Feature engineering for target match
│   ├── 03_bayesian_inference.ipynb      # PyMC model & MCMC sampling
│   └── 04_posterior_analysis.ipynb      # Visualizations
│
│
└── README.md
````

-----

##  Methodology

### 1\. Prior Elicitation (Notebook 01)

We construct **informative priors** based on 2023 historical performance at Wimbledon.

  * **Model:** Beta Distribution $\theta \sim \text{Beta}(\alpha, \beta)$
  * **Variance Inflation:** Historical counts ($n$) are divided by a factor of 10. This creates "weakly informative" priors that guide the model but allow the specific 2024 match data to dominate the posterior as the game progresses.

### 2\. Observation Encoding (Notebook 02)

Raw point-by-point data is transformed into binary observation arrays suitable for Bernoulli likelihoods:

  * **Serve Outcome:** $y_{srv} \in \{0, 1\}$ (Fault / In)
  * **Ace Outcome:** $y_{ace} \in \{0, 1\}$ (Not Ace / Ace)
  * **Server Index:** $idx \in \{0, 1\}$ (Sinner / Medvedev)

### 3\. Bayesian Inference (Notebook 03)

We implement a probabilistic model using **PyMC**.

  * **Likelihood:** Bernoulli process for each point.
  * **Inference:** NUTS (No-U-Turn Sampler) to approximate the posterior distributions.

-----

## Usage Guide

To reproduce the analysis, execute the notebooks in numerical order:

**1. `01_Prior_Elicitation.ipynb`**

  * **Input:** Raw CSV files in `data/raw/`.
  * **Action:** Fixes missing serve data logic; calculates Beta parameters for 2023.
  * **Output:** `priors_2023.pkl` and corrected CSVs.

**2. `02_Match_Data_Engineering.ipynb`**

  * **Input:** Corrected 2024 CSV.
  * **Action:** Filters Match 1501; encodes rolling features and binary targets.
  * **Output:** `match_1501_features.pkl`.

**3. `03_Bayesian_Inference.ipynb`**

  * **Input:** Pickle files from previous steps.
  * **Action:** Defines and runs the PyMC model.
  * **Output:** Inference Trace (NetCDF format).

-----

##  Data Source

This project uses the **Grand Slam Point-by-Point Data** maintained by Jeff Sackmann.

  * Repository: [https://github.com/JeffSackmann/tennis\_slam\_pointbypoint](https://github.com/JeffSackmann/tennis_slam_pointbypoint)

