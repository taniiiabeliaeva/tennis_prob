import math

"""
TENNIS SCORING LOGIC & HIERARCHY
--------------------------------
This module implements the hierarchical scoring rules of tennis to project 
a latent skill parameter (theta) onto a match outcome.

Assumption: 'theta' represents the player's DOMINANCE (Probability of winning ANY point).
- theta = 0.50: Even match.
- theta = 0.52: Slight edge.
- theta = 0.60: Crushing victory.
"""

def prob_game(p):
    """
    calculates the probability of winning a standard game given 
    probability 'p' of winning a single point.
    """
    # perfect win (4-0)
    prob_4_0 = p**4
    
    # win to 15 (4-1)
    prob_4_1 = 4 * (p**4) * (1-p)
    
    # win to 30 (4-2)
    prob_4_2 = 10 * (p**4) * ((1-p)**2)
    
    # reach deuce (3-3)
    prob_reach_deuce = 20 * (p**3) * ((1-p)**3)
    
    # win from deuce
    win_from_deuce = p**2 / (p**2 + (1-p)**2)
    
    return prob_4_0 + prob_4_1 + prob_4_2 + (prob_reach_deuce * win_from_deuce)

def prob_tiebreak(p):
    """
    calculates probability of winning a tiebreak (first to 7, win by 2).
    """
    prob_win = 0
    
    # scenario a: win before reaching 6-6
    for k in range(6):
        points_played = 6 + k
        combinations = math.comb(points_played, k)
        prob_win += combinations * (p**7) * ((1-p)**k)
        
    # scenario b: reach 6-6
    prob_reach_66 = math.comb(12, 6) * (p**6) * ((1-p)**6)
    
    # win from 6-6
    win_from_66 = p**2 / (p**2 + (1-p)**2)
    
    return prob_win + (prob_reach_66 * win_from_66)

def prob_set(p):
    """
    calculates probability of winning a standard set.
    Assumes 'p' is the probability of winning ANY game (Dominance). this change help us to avoide clonning problem
    """
    # probability of winning a single game
    p_game = prob_game(p)
    
    # 1. win before tiebreak (6-0 to 6-4)
    prob_win_easy = 0
    for k in range(5):
        # opponent wins k games
        prob_win_easy += math.comb(5+k, k) * (p_game**6) * ((1-p_game)**k)
        
    # 2. win 7-5
    # reach 5-5 (10 games, 5 wins each)
    prob_5_5 = math.comb(10, 5) * (p_game**5) * ((1-p_game)**5)
    # win next 2 games
    prob_7_5 = prob_5_5 * p_game * p_game 
    
    # 3. win in tiebreak (7-6)
    # reach 5-5 -> split 1-1 -> reach 6-6
    prob_reach_66 = prob_5_5 * 2 * p_game * (1-p_game)
    
    # tiebreak depends on point probability 'p'
    prob_win_tb = prob_tiebreak(p)
    
    return prob_win_easy + prob_7_5 + (prob_reach_66 * prob_win_tb)

def prob_match_win(theta):
    """
    THE PROJECTOR FUNCTION
    ----------------------
    input: theta (probability of winning ANY point)
    output: probability of winning a best-of-5 match
    """
    p_s = prob_set(theta)
    
    # best of 5 logic
    win_3_0 = p_s**3
    win_3_1 = 3 * (p_s**3) * (1 - p_s)
    win_3_2 = 6 * (p_s**3) * ((1 - p_s)**2)
    
    return win_3_0 + win_3_1 + win_3_2


#this code below can be performed as a test of a function
#from tennis_utils import prob_match_win
#
#print(f"{'Skill (Point)':<15} | {'Win Probability (Match)':<25}")
#print("-" * 45)
#
# Check values from 0.50 to 0.60 
#for p in np.linspace(0.50, 0.60, 6):
#    win_prob = prob_match_win(p)
#    print(f"{p:.2f}            | {win_prob:.4f} ({win_prob:.1%})")
#
