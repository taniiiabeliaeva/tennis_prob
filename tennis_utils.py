import numpy as np

def prob_game(p):
    """
    Calculates the probability of winning a game given the probability 'p' of winning a point.
    Includes the Deuce logic (infinite series summation).
    
    Formula: P(Game) = P(Win < Deuce) + P(Reach Deuce) * P(Win | Deuce)
    """
    # 1. Win before Deuce (4-0, 4-1, 4-2)
    # p^4 + 4p^4(1-p) + 10p^4(1-p)^2
    prob_4_0 = p**4
    prob_4_1 = 4 * (p**4) * (1-p)
    prob_4_2 = 10 * (p**4) * ((1-p)**2)
    
    # 2. Reach Deuce (3-3)
    # 20 * p^3 * (1-p)^3
    prob_deuce = 20 * (p**3) * ((1-p)**3)
    
    # 3. Win from Deuce
    # Win 2 consecutive points: p^2
    # Probability of returning to Deuce: 2p(1-p)
    # Geometric series sum: p^2 / (1 - 2p(1-p)) = p^2 / (p^2 + (1-p)^2)
    prob_win_deuce = p**2 / (p**2 + (1-p)**2)
    
    return prob_4_0 + prob_4_1 + prob_4_2 + (prob_deuce * prob_win_deuce)

def prob_tiebreak(p):
    """
    Calculates probability of winning a 7-point tiebreak.
    First to 7, win by 2.
    """
    # P(Win before 6-6)
    # Scores: 7-0, 7-1, 7-2, 7-3, 7-4, 7-5
    # For score 7-k, you need 6 wins in first 6+k points, then win the last one.
    # Combinations: (6+k) choose k
    prob_pre_66 = 0
    for k in range(6): # Opponent scores 0 to 5
        combs = np.math.comb(6+k, k)
        prob_pre_66 += combs * (p**7) * ((1-p)**k)
        
    # P(Reach 6-6)
    # 12 points played, 6 won each: (12 choose 6) * p^6 * (1-p)^6
    prob_reach_66 = np.math.comb(12, 6) * (p**6) * ((1-p)**6)
    
    # P(Win from 6-6)
    # Same logic as Deuce: Win 2 in a row
    prob_win_66 = p**2 / (p**2 + (1-p)**2)
    
    return prob_pre_66 + (prob_reach_66 * prob_win_66)

def prob_set(p_serve):
    """
    Calculates probability of winning a standard Tiebreak Set (Best of 6 games).
    Assumes serving player has advantage 'p_serve' and receiving player has disadvantage '1-p_serve'.
    """
    # 1. Calculate P(Hold Serve) and P(Break Opponent)
    p_hold = prob_game(p_serve)
    p_break = prob_game(1 - p_serve) # Assuming opponent has same skill p_serve
    
    # Simplified assumption for 50h project:
    # We approximate P(Win Game) as the average of Holding and Breaking
    # This assumes we play an even number of service games.
    p_win_game = (p_hold + p_break) / 2
    
    # 2. Probability of winning set without tiebreak (6-0 to 6-4, or 7-5)
    # This is complex to model exactly game-by-game without simulation.
    # For this project, we use the "Binomial Approximation" which is standard in sports analytics
    # unless you want to write a full Markov Chain for the set (overkill).
    
    # However, since you are Student A and want to impress, let's use the 
    # slightly better "Set Markov Chain" approximation:
    
    # P(Win Set) ~ P(Win 6 games) but we can approximate it as winning a "Race to 6"
    # Using the same logic as prob_game but to 6.
    
    # Let w = p_win_game
    w = p_win_game
    
    # Win 6-0 to 6-4
    prob_pre_tiebreak = 0
    for k in range(5): # Opponent wins 0 to 4 games
        combs = np.math.comb(5+k, k)
        prob_pre_tiebreak += combs * (w**6) * ((1-w)**k)
        
    # Win 7-5 (Must reach 5-5, then win 2)
    prob_5_5 = np.math.comb(10, 5) * (w**5) * ((1-w)**5)
    prob_7_5 = prob_5_5 * w * w # Win next 2
    
    # Reach Tiebreak (6-6) -> (Must reach 5-5, then split 1-1)
    # Path to 6-6: Reach 5-5, then (Win,Lose) or (Lose,Win)
    prob_reach_66 = prob_5_5 * 2 * w * (1-w)
    
    # Win Tiebreak
    # For TB, we usually assume average serve probability again
    p_tb = prob_tiebreak(p_serve) # Or (p_serve + (1-p_serve))/2 = 0.5?
    # Better: Use the raw point prob 'p_serve' for the TB calculation
    
    return prob_pre_tiebreak + prob_7_5 + (prob_reach_66 * p_tb)

def prob_match_win(theta):
    """
    The Main Projector Function.
    Input: theta (Probability of winning a point on serve, e.g., 0.65)
    Output: Probability of winning a Best-of-5 Match
    """
    # 1. Get P(Set)
    p_set = prob_set(theta)
    
    # 2. Get P(Match) - Best of 5 (First to 3 sets)
    # Win 3-0, 3-1, 3-2
    w = p_set
    l = 1 - p_set
    
    prob_3_0 = w**3
    prob_3_1 = 3 * (w**3) * l  # (WWWL, WWLW, WLWW) -> 3 combinations
    prob_3_2 = 6 * (w**3) * (l**2) # (WWWLL... permutations ending in W) -> 6 combinations
    
    return prob_3_0 + prob_3_1 + prob_3_2