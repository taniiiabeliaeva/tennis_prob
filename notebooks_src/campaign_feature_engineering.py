"""
Campaign Feature Engineering for Transfer Learning
===================================================

This script aggregates Carlos Alcaraz's service points across all Wimbledon 2023+2024
matches to prepare data for the transfer learning approach.

Output:
- alcaraz_training_2023_2024.pkl: 13 matches (~1,400 points) for Stage 1
- alcaraz_target_2024_final.pkl: 1 match (~150 points) for Stage 2
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Target player
PLAYER_NAME = 'Carlos Alcaraz'

# Target match for Stage 2 (2024 Final vs Djokovic)
TARGET_MATCH_ID = '2024-wimbledon-1701'  # Will verify this below

# Feature engineering constants
MAX_RALLY_LENGTH = 30.0


def load_all_data():
    """Load all match and point data from 2023 and 2024."""
    # Load matches
    matches_2023 = pd.read_csv(DATA_DIR / 'raw' / '2023-wimbledon-matches.csv')
    matches_2024 = pd.read_csv(DATA_DIR / 'raw' / '2024-wimbledon-matches.csv')
    
    # Load points
    points_2023 = pd.read_csv(PROCESSED_DIR / '2023-wimbledon-points-corrected.csv')
    points_2024 = pd.read_csv(PROCESSED_DIR / '2024-wimbledon-points-corrected.csv')
    
    # Combine
    matches_all = pd.concat([matches_2023, matches_2024], ignore_index=True)
    points_all = pd.concat([points_2023, points_2024], ignore_index=True)
    
    return matches_all, points_all


def find_player_matches(matches_df, player_name):
    """Find all matches for a player, returning match_id and whether they're P1 or P2."""
    player_matches = []
    
    for _, row in matches_df.iterrows():
        if player_name in row['player1']:
            player_matches.append({
                'match_id': row['match_id'],
                'is_p1': True,
                'opponent': row['player2']
            })
        elif player_name in row['player2']:
            player_matches.append({
                'match_id': row['match_id'],
                'is_p1': False,
                'opponent': row['player1']
            })
    
    return pd.DataFrame(player_matches)


def filter_player_serving(points_df, match_id, is_p1):
    """
    Filter for player's service points.
    
    When player is P1: PointServer=1 means they're serving
    When player is P2: PointServer=2 means they're serving
    """
    match_points = points_df[points_df['match_id'] == match_id].copy()
    
    if is_p1:
        # Player is P1, their serves are PointServer=1
        serve_points = match_points[match_points['PointServer'] == 1].copy()
        # Win means PointWinner=1
        serve_points['y'] = (serve_points['PointWinner'] == 1).astype(int)
    else:
        # Player is P2, their serves are PointServer=2
        serve_points = match_points[match_points['PointServer'] == 2].copy()
        # Win means PointWinner=2
        serve_points['y'] = (serve_points['PointWinner'] == 2).astype(int)
    
    # Remove points where winner wasn't tracked
    serve_points = serve_points[serve_points['PointWinner'] != 0]
    
    return serve_points.reset_index(drop=True)


def engineer_features(df, is_p1):
    """Apply all feature engineering to a match DataFrame."""
    
    # --- Feature A: Second Serve ---
    # ServeNumber: 1 = first serve, 2 = second serve
    df['is_second_serve'] = (df['ServeNumber'] == 2).astype(int)
    
    # --- Feature B: Depth Pressure (Opponent's deep returns) ---
    df['ReturnDepth'] = df['ReturnDepth'].fillna('ND')
    df['depth_pressure'] = (df['ReturnDepth'] == 'D').astype(int)
    
    # --- Feature C: Previous Rally Length ---
    df['RallyCount'] = df['RallyCount'].fillna(0)
    df['prev_rally_length_raw'] = df['RallyCount'].shift(1).fillna(0)
    df['prev_rally_length'] = (df['prev_rally_length_raw'] / MAX_RALLY_LENGTH).clip(0, 1)
    
    # --- Feature D: Aggression (Wide serve or CTL) ---
    df['ServeDepth'] = df['ServeDepth'].fillna('NCTL')
    df['ServeWidth'] = df['ServeWidth'].fillna('C')
    df['is_aggressive'] = (
        (df['ServeWidth'] == 'W') | 
        (df['ServeDepth'] == 'CTL')
    ).astype(int)
    
    # --- Feature E: Pressure (Break points / Deuce) ---
    # Need to handle P1/P2 perspective for break points
    if is_p1:
        is_breakpoint = (df['P1BreakPoint'] == 1) | (df['P2BreakPoint'] == 1)
    else:
        is_breakpoint = (df['P1BreakPoint'] == 1) | (df['P2BreakPoint'] == 1)
    
    is_deuce = (df['P1Score'] == '40') & (df['P2Score'] == '40')
    df['is_pressure'] = (is_breakpoint | is_deuce).astype(int)
    
    return df


def verify_features(df):
    """Verify no NaNs in feature columns."""
    feature_cols = ['is_second_serve', 'depth_pressure', 'prev_rally_length', 
                    'is_aggressive', 'is_pressure', 'y']
    nan_count = df[feature_cols].isna().sum().sum()
    assert nan_count == 0, f"Found {nan_count} NaNs in features!"


def export_for_pymc(df, output_path, include_match_info=False):
    """Export data in format ready for PyMC modeling."""
    model_data = {
        'y_obs': df['y'].values,
        'second_serve_obs': df['is_second_serve'].values,
        'depth_pressure_obs': df['depth_pressure'].values,
        'prev_rally_obs': df['prev_rally_length'].values,
        'aggr_obs': df['is_aggressive'].values,
        'pressure_obs': df['is_pressure'].values,
        'time_idx': np.arange(len(df))
    }
    
    if include_match_info:
        model_data['match_id'] = df['match_id'].values
        model_data['match_code'] = df['match_code'].values if 'match_code' in df.columns else None
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data


def main():
    """Main function to run the campaign feature engineering pipeline."""
    print("=" * 60)
    print("CAMPAIGN FEATURE ENGINEERING")
    print(f"Player: {PLAYER_NAME}")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading 2023+2024 data...")
    matches_all, points_all = load_all_data()
    print(f"   Loaded {len(matches_all)} matches, {len(points_all)} points")
    
    # Find player matches
    print(f"\n2. Finding {PLAYER_NAME} matches...")
    player_matches = find_player_matches(matches_all, PLAYER_NAME)
    print(f"   Found {len(player_matches)} matches:")
    for _, row in player_matches.iterrows():
        print(f"     {row['match_id']}: vs {row['opponent']} ({'P1' if row['is_p1'] else 'P2'})")
    
    # Identify target match (latest match = Final)
    target_match_id = player_matches.iloc[-1]['match_id']  # Last match is the Final
    print(f"\n   Target match (Stage 2): {target_match_id}")
    
    # Process all matches
    print("\n3. Processing matches...")
    all_serve_points = []
    
    for _, match_info in player_matches.iterrows():
        m_id = match_info['match_id']
        is_p1 = match_info['is_p1']
        
        # Filter for player's serves
        serve_pts = filter_player_serving(points_all, m_id, is_p1)
        
        if len(serve_pts) == 0:
            print(f"     ⚠ No points for {m_id}")
            continue
        
        # Engineer features
        serve_pts = engineer_features(serve_pts, is_p1)
        serve_pts['match_id'] = m_id
        
        print(f"     {m_id}: {len(serve_pts)} service points, win rate: {serve_pts['y'].mean():.1%}")
        all_serve_points.append(serve_pts)
    
    # Combine all
    df_all = pd.concat(all_serve_points, ignore_index=True)
    
    # Verify features
    print("\n4. Verifying features...")
    verify_features(df_all)
    print("   ✓ No NaNs in features")
    
    # Split into training and target
    print("\n5. Splitting into training and target...")
    df_training = df_all[df_all['match_id'] != target_match_id].copy()
    df_target = df_all[df_all['match_id'] == target_match_id].copy()
    
    # Add match codes for training (for potential hierarchical modeling)
    df_training['match_code'] = pd.Categorical(df_training['match_id']).codes
    
    print(f"   Training: {len(df_training)} points from {df_training['match_id'].nunique()} matches")
    print(f"   Target: {len(df_target)} points from 1 match")
    
    # Export
    print("\n6. Exporting for PyMC...")
    
    training_path = PROCESSED_DIR / 'alcaraz_training_2023_2024.pkl'
    target_path = PROCESSED_DIR / 'alcaraz_target_2024_final.pkl'
    
    export_for_pymc(df_training, training_path, include_match_info=True)
    export_for_pymc(df_target, target_path)
    
    print(f"   Training data: {training_path}")
    print(f"   Target data: {target_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    print(f"\nTRAINING DATA ({len(df_training)} points):")
    print(f"  Win rate: {df_training['y'].mean():.1%}")
    print(f"  Second serve rate: {df_training['is_second_serve'].mean():.1%}")
    print(f"  Deep return rate: {df_training['depth_pressure'].mean():.1%}")
    print(f"  Aggression rate: {df_training['is_aggressive'].mean():.1%}")
    print(f"  Pressure rate: {df_training['is_pressure'].mean():.1%}")
    
    print(f"\nTARGET DATA ({len(df_target)} points):")
    print(f"  Win rate: {df_target['y'].mean():.1%}")
    print(f"  Second serve rate: {df_target['is_second_serve'].mean():.1%}")
    print(f"  Deep return rate: {df_target['depth_pressure'].mean():.1%}")
    print(f"  Aggression rate: {df_target['is_aggressive'].mean():.1%}")
    print(f"  Pressure rate: {df_target['is_pressure'].mean():.1%}")
    
    return df_training, df_target


if __name__ == "__main__":
    df_training, df_target = main()
