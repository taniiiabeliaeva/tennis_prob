"""
Multimodal State-Space Model Feature Engineering
=================================================

This script creates features for the Gaussian Random Walk State-Space Model
that captures player momentum, tactical state, and contextual factors.

Features (Updated v2 - Replacing inconclusive fatigue with stronger predictors):
1. is_second_serve (Tactical State): The penalty for missing the first serve
2. depth_pressure (Opponent Pressure): Did Medvedev hit deep returns?
3. prev_rally_length (Immediate Load): Was the previous rally exhausting?
4. is_aggressive (Tactical State): Did the server aim for the lines?
5. is_pressure (Mental State): Is it a "Big Point"?

We model Sinner's Service Games (PointServer == 1) to isolate HIS momentum dynamics.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Constants
TARGET_MATCH_ID = '2024-wimbledon-1501'
PLAYER_1_NAME = 'Jannik Sinner'
PLAYER_2_NAME = 'Daniil Medvedev'
MAX_RALLY_LENGTH = 30.0  # For normalizing rally length


def load_data():
    """Load the match data."""
    points_2024 = pd.read_csv(PROCESSED_DIR / '2024-wimbledon-points-corrected.csv')
    return points_2024


def filter_sinner_serving(df):
    """Filter for Sinner serving points only."""
    mask = (df['match_id'] == TARGET_MATCH_ID) & (df['PointServer'] == 1)
    df_sinner = df[mask].copy().reset_index(drop=True)
    
    # Drop rare rows where winner wasn't tracked (PointWinner == 0)
    df_sinner = df_sinner[df_sinner['PointWinner'] != 0]
    
    # Create Binary Target: 1 if Sinner (Server) Won, 0 if Lost
    df_sinner['y'] = (df_sinner['PointWinner'] == 1).astype(int)
    
    return df_sinner


def create_second_serve_feature(df):
    """
    Feature A: The "Second Serve" Penalty (Tactical State)
    
    Binary indicator: 1 if this is a second serve, 0 if first serve.
    
    Hypothesis: The drop from 1st to 2nd serve is massive (~15-20% win prob).
    Even if Sinner's momentum (alpha_t) is high, missing the first serve
    immediately puts him at a disadvantage.
    """
    # ServeNumber: 1 = first serve, 2 = second serve, 0 = unknown/missing
    df['is_second_serve'] = (df['ServeNumber'] == 2).astype(int)
    return df


def create_depth_pressure_feature(df):
    """
    Feature B: The "Deep Return" Interaction (Opponent Pressure)
    
    Binary indicator: 1 if Medvedev's return was Deep (D), 0 otherwise.
    
    Hypothesis: Sinner loses points when Medvedev hits deep returns.
    This captures Medvedev's defensive quality - a factor EXTERNAL to Sinner.
    """
    # ReturnDepth: 'D' = Deep, 'ND' = Not Deep
    df['ReturnDepth'] = df['ReturnDepth'].fillna('ND')
    df['depth_pressure'] = (df['ReturnDepth'] == 'D').astype(int)
    return df


def create_prev_rally_length_feature(df):
    """
    Feature C: The "Lung-Buster" Effect (Immediate Physical Load)
    
    Scalar: The rally length of the PREVIOUS point, normalized to 0-1.
    
    Hypothesis: If the previous point had 15+ shots, the server's legs
    are heavy NOW. This captures immediate fatigue rather than cumulative.
    """
    # Shift RallyCount to get previous rally length
    df['RallyCount'] = df['RallyCount'].fillna(0)
    df['prev_rally_length_raw'] = df['RallyCount'].shift(1).fillna(0)
    
    # Normalize to 0-1 range (cap at MAX_RALLY_LENGTH)
    df['prev_rally_length'] = (df['prev_rally_length_raw'] / MAX_RALLY_LENGTH).clip(0, 1)
    return df


def create_aggression_feature(df):
    """
    Feature B: Tactical State (Aggression)
    
    Did the server aim for the lines? (Wide serve or Close to Line)
    """
    # Handle NaNs by filling with 'Safe' options
    df['ServeDepth'] = df['ServeDepth'].fillna('NCTL')
    df['ServeWidth'] = df['ServeWidth'].fillna('C')
    
    # Definition of Aggression: Serving Wide (W) OR Close To Line (CTL)
    df['is_aggressive'] = (
        (df['ServeWidth'] == 'W') | 
        (df['ServeDepth'] == 'CTL')
    ).astype(int)
    
    return df


def create_pressure_feature(df):
    """
    Feature C: Mental State (Pressure)
    
    Is it a "Big Point"? (Break Points or Deuce)
    """
    is_breakpoint = (df['P1BreakPoint'] == 1) | (df['P2BreakPoint'] == 1)
    is_deuce = (df['P1Score'] == '40') & (df['P2Score'] == '40')
    df['is_pressure'] = (is_breakpoint | is_deuce).astype(int)
    
    return df


def verify_features(df):
    """Verify that all features are valid (no NaNs)."""
    feature_cols = ['is_second_serve', 'depth_pressure', 'prev_rally_length', 
                    'is_aggressive', 'is_pressure']
    nan_count = df[feature_cols].isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaNs in regressors!"
    print("✓ No NaNs in regressors")


def export_for_pymc(df, output_path):
    """Export data in format ready for PyMC modeling."""
    model_data = {
        'y_obs': df['y'].values,
        # New features (v2)
        'second_serve_obs': df['is_second_serve'].values,
        'depth_pressure_obs': df['depth_pressure'].values,
        'prev_rally_obs': df['prev_rally_length'].values,
        # Existing features
        'aggr_obs': df['is_aggressive'].values,
        'pressure_obs': df['is_pressure'].values,
        'time_idx': np.arange(len(df))
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data


def main():
    """Main function to run the feature engineering pipeline."""
    print("=" * 60)
    print("MULTIMODAL STATE-SPACE MODEL FEATURE ENGINEERING (v2)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    points_2024 = load_data()
    print(f"   Loaded {len(points_2024)} rows")
    
    # Filter for Sinner serving
    print("\n2. Filtering for Sinner serving points...")
    df_sinner = filter_sinner_serving(points_2024)
    print(f"   Sinner serving points: {len(df_sinner)}")
    print(f"   Win rate: {df_sinner['y'].mean():.3f}")
    
    # Create NEW features (replacing fatigue)
    print("\n3. Creating 'Second Serve' feature (Tactical State)...")
    df_sinner = create_second_serve_feature(df_sinner)
    print(f"   Second serves: {df_sinner['is_second_serve'].sum()} / {len(df_sinner)} ({df_sinner['is_second_serve'].mean():.1%})")
    
    print("\n4. Creating 'Depth Pressure' feature (Opponent Pressure)...")
    df_sinner = create_depth_pressure_feature(df_sinner)
    print(f"   Deep returns: {df_sinner['depth_pressure'].sum()} / {len(df_sinner)} ({df_sinner['depth_pressure'].mean():.1%})")
    
    print("\n5. Creating 'Previous Rally Length' feature (Immediate Load)...")
    df_sinner = create_prev_rally_length_feature(df_sinner)
    print(f"   Prev rally range: [{df_sinner['prev_rally_length'].min():.2f}, {df_sinner['prev_rally_length'].max():.2f}]")
    print(f"   Prev rally mean: {df_sinner['prev_rally_length'].mean():.3f}")
    
    print("\n6. Creating 'Aggression' feature (Tactical State)...")
    df_sinner = create_aggression_feature(df_sinner)
    print(f"   Aggressive serves: {df_sinner['is_aggressive'].sum()} / {len(df_sinner)} ({df_sinner['is_aggressive'].mean():.1%})")
    
    print("\n7. Creating 'Pressure' feature (Mental State)...")
    df_sinner = create_pressure_feature(df_sinner)
    print(f"   Pressure points: {df_sinner['is_pressure'].sum()} / {len(df_sinner)} ({df_sinner['is_pressure'].mean():.1%})")
    
    # Verify features
    print("\n8. Verifying features...")
    verify_features(df_sinner)
    
    # Export for PyMC
    print("\n9. Exporting for PyMC...")
    output_path = PROCESSED_DIR / 'sinner_features_2024.pkl'
    model_data = export_for_pymc(df_sinner, output_path)
    print(f"   Data exported to: {output_path}")
    print(f"   Data Prepared: {len(df_sinner)} points for Sinner Service Model.")
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY (v2)")
    print("=" * 60)
    print(f"Total points: {len(df_sinner)}")
    print(f"Win rate: {df_sinner['y'].mean():.3f}")
    print(f"")
    print(f"NEW FEATURES:")
    print(f"  Second serve rate: {df_sinner['is_second_serve'].mean():.1%}")
    print(f"  Deep return rate: {df_sinner['depth_pressure'].mean():.1%}")
    print(f"  Prev rally mean: {df_sinner['prev_rally_length'].mean():.3f}")
    print(f"")
    print(f"EXISTING FEATURES:")
    print(f"  Aggression rate: {df_sinner['is_aggressive'].mean():.1%}")
    print(f"  Pressure rate: {df_sinner['is_pressure'].mean():.1%}")
    
    return model_data


if __name__ == "__main__":
    main()
