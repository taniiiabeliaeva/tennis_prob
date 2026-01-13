"""
Match Probability Calculator
============================

This module provides analytical functions for calculating tennis match 
probabilities from point-level win probabilities.

Adapted from: external/tennis_bayes_point_based-master/winning_prob.py

The key insight from the external project is that given the probability of
winning a point on serve (rally_win_prob), we can analytically compute:
1. Probability of holding serve
2. Probability of winning a game
3. Probability of winning a tiebreak
4. Probability of winning a set
5. Probability of winning a match

This allows us to translate our Bayesian model's point-level predictions
into match-level win probabilities.

Usage:
    from match_probability import MatchProbabilityCalculator
    
    calc = MatchProbabilityCalculator()
    match_prob = calc.win_probability_match(p_serve_a=0.65, p_serve_b=0.60)
"""

import numpy as np
from functools import lru_cache
from scipy.special import expit  # Logistic/sigmoid function


class MatchProbabilityCalculator:
    """
    Calculate match win probabilities from serve point win probabilities.
    
    This uses the recursive analytical formulas from the external Bayesian
    tennis model to compute exact probabilities.
    """
    
    def __init__(self, best_of_five: bool = True):
        """
        Initialize the calculator.
        
        Args:
            best_of_five: If True, calculate for best-of-5 format (Grand Slam).
                         If False, use best-of-3 format.
        """
        self.best_of_five = best_of_five
    
    def hold_serve_prob(self, rally_win_prob: float) -> float:
        """
        Calculate the probability of holding serve.
        
        Given the probability of winning a point on serve, this calculates
        the probability of winning the game (holding serve).
        
        The formula accounts for:
        - Winning in 4 points (game, no deuce)
        - Winning after deuce situations
        
        Args:
            rally_win_prob: Probability of winning a point when serving
        
        Returns:
            Probability of holding serve (winning the game)
        """
        p = rally_win_prob
        q = 1 - p
        
        # Probability of winning in 4, 5, or 6 points (before deuce)
        term_1 = p**4 * (1 + 4*q + 10*q**2)
        
        # Probability of reaching deuce and then winning
        # Deuce is reached, then must win 2 in a row out of 2
        term_2 = 20 * (p**3) * (q**3) * (p**2) / (1 - 2*p*q)
        
        return term_1 + term_2
    
    @lru_cache(maxsize=1024)
    def _prob_reach_tiebreak_score(self, i: int, j: int, p_a: float, p_b: float) -> float:
        """
        Recursive calculation of reaching a tiebreak score.
        
        Args:
            i: Player A's tiebreak score
            j: Player B's tiebreak score
            p_a: Probability player A wins a point on their serve
            p_b: Probability player B wins a point on their serve
        
        Returns:
            Probability of reaching tiebreak score [i, j]
        """
        q_a = 1 - p_a
        q_b = 1 - p_b
        
        # Initial conditions
        if i == 0 and j == 0:
            return 1.0
        if i < 0 or j < 0:
            return 0.0
        
        # Determine who served last point
        # In tiebreak: A serves first, then alternates every 2 points
        a_served_last = (((i - 1 + j) % 4 == 0) or ((i - 1 + j) % 4 == 3))
        
        total = 0.0
        
        if a_served_last:
            # A served last, so current point came from A's serve
            if not (i == 7 and j <= 6):  # Don't go beyond winning score
                total += self._prob_reach_tiebreak_score(i, j-1, p_a, p_b) * q_a
            if not (j == 7 and i <= 6):
                total += self._prob_reach_tiebreak_score(i-1, j, p_a, p_b) * p_a
        else:
            # B served last
            if not (i == 7 and j <= 6):
                total += self._prob_reach_tiebreak_score(i, j-1, p_a, p_b) * p_b
            if not (j == 7 and i <= 6):
                total += self._prob_reach_tiebreak_score(i-1, j, p_a, p_b) * q_b
        
        return total
    
    def prob_win_tiebreak_a(self, p_a: float, p_b: float) -> float:
        """
        Calculate probability that player A wins the tiebreak.
        
        Args:
            p_a: Probability player A wins a point on their serve
            p_b: Probability player B wins a point on their serve
        
        Returns:
            Probability player A wins the tiebreak
        """
        q_a = 1 - p_a
        q_b = 1 - p_b
        
        total = 0.0
        
        # Win 7-0, 7-1, 7-2, 7-3, 7-4, 7-5
        for j in range(6):
            total += self._prob_reach_tiebreak_score(7, j, p_a, p_b)
        
        # Win after reaching 6-6 (must win 2 straight with alternating serves)
        reach_66 = self._prob_reach_tiebreak_score(6, 6, p_a, p_b)
        prob_win_from_66 = (p_a * q_b) / (1 - p_a*p_b - q_a*q_b)
        total += reach_66 * prob_win_from_66
        
        return total
    
    @lru_cache(maxsize=1024)
    def _prob_reach_set_score(self, i: int, j: int, p_a: float, p_b: float) -> float:
        """
        Recursive calculation of reaching a set score.
        
        Args:
            i: Player A's games won in set
            j: Player B's games won in set
            p_a: Probability player A wins a point on their serve
            p_b: Probability player B wins a point on their serve
        
        Returns:
            Probability of reaching set score [i, j]
        """
        # Validate score
        valid = (j <= 6 and i <= 6) or (i == 7 and j <= 6) or (i <= 6 and j == 7)
        if not valid:
            return 0.0
        
        hold_a = self.hold_serve_prob(p_a)
        hold_b = self.hold_serve_prob(p_b)
        break_a = 1 - hold_a
        break_b = 1 - hold_b
        
        # Determine who served last
        a_served_last = ((i - 1 + j) % 2 == 0)
        
        # Initial conditions
        if i == 0 and j == 0:
            return 1.0
        if i < 0 or j < 0:
            return 0.0
        
        # Tiebreak cases
        if i == 6 and j == 7:
            reach_66 = self._prob_reach_set_score(6, 6, p_a, p_b)
            return reach_66 * (1 - self.prob_win_tiebreak_a(p_a, p_b))
        
        if i == 7 and j == 6:
            reach_66 = self._prob_reach_set_score(6, 6, p_a, p_b)
            return reach_66 * self.prob_win_tiebreak_a(p_a, p_b)
        
        # 7-5 case (break at 6-5)
        if i == 7 and j == 5:
            return self._prob_reach_set_score(6, 5, p_a, p_b) * break_b
        if i == 5 and j == 7:
            return self._prob_reach_set_score(5, 6, p_a, p_b) * hold_b
        
        total = 0.0
        
        if a_served_last:
            if not (j == 6 and i <= 5):
                total += self._prob_reach_set_score(i-1, j, p_a, p_b) * hold_a
            if not (i == 6 and j <= 5):
                total += self._prob_reach_set_score(i, j-1, p_a, p_b) * break_a
        else:
            if not (j == 6 and i <= 5):
                total += self._prob_reach_set_score(i-1, j, p_a, p_b) * break_b
            if not (i == 6 and j <= 5):
                total += self._prob_reach_set_score(i, j-1, p_a, p_b) * hold_b
        
        return total
    
    def prob_win_set_a(self, p_a: float, p_b: float) -> float:
        """
        Calculate probability that player A wins a set.
        
        Args:
            p_a: Probability player A wins a point on their serve
            p_b: Probability player B wins a point on their serve
        
        Returns:
            Probability player A wins the set
        """
        total = 0.0
        
        # Win 6-0, 6-1, 6-2, 6-3, 6-4
        for j in range(5):
            total += self._prob_reach_set_score(6, j, p_a, p_b)
        
        # Win 7-5
        total += self._prob_reach_set_score(7, 5, p_a, p_b)
        
        # Win tiebreak (7-6)
        total += self._prob_reach_set_score(7, 6, p_a, p_b)
        
        return total
    
    def prob_win_match_a(self, p_a: float, p_b: float) -> float:
        """
        Calculate probability that player A wins the match.
        
        Args:
            p_a: Probability player A wins a point on their serve
            p_b: Probability player B wins a point on their serve
        
        Returns:
            Probability player A wins the match
        """
        # Clear cache for new calculation with different probabilities
        self._prob_reach_tiebreak_score.cache_clear()
        self._prob_reach_set_score.cache_clear()
        
        prob_a_win_set = self.prob_win_set_a(p_a, p_b)
        prob_b_win_set = self.prob_win_set_a(p_b, p_a)
        
        if not self.best_of_five:
            # Best of 3: need to win 2 sets
            # Win 2-0 or 2-1
            return prob_a_win_set**2 + 2 * prob_a_win_set**2 * prob_b_win_set
        else:
            # Best of 5: need to win 3 sets
            # Win 3-0, 3-1, or 3-2
            p3 = prob_a_win_set**3
            return p3 + 3*p3*prob_b_win_set + 6*p3*prob_b_win_set**2
    
    def win_probability_from_logit(self, logit_p: np.ndarray) -> np.ndarray:
        """
        Convert logit (log-odds) to win probability.
        
        Args:
            logit_p: Array of logit values from the Bayesian model
        
        Returns:
            Array of win probabilities (0 to 1)
        """
        return expit(logit_p)
    
    def calculate_match_win_trajectory(self, 
                                       point_win_probs_a: np.ndarray,
                                       point_win_probs_b: np.ndarray,
                                       server_sequence: np.ndarray) -> np.ndarray:
        """
        Calculate the evolving match win probability throughout a match.
        
        This is the key function for Phase 5 visualization - it shows how
        the probability of winning the match evolves as points are played.
        
        Args:
            point_win_probs_a: Win probability for each point when A serves
            point_win_probs_b: Win probability for each point when B serves
            server_sequence: Who served each point (0=A, 1=B)
        
        Returns:
            Array of match win probabilities at each point
        """
        n_points = len(server_sequence)
        match_win_probs = np.zeros(n_points)
        
        for i in range(n_points):
            # Use running average of point win probabilities up to this point
            # This gives a smoother trajectory
            if i == 0:
                p_a = point_win_probs_a[0]
                p_b = point_win_probs_b[0]
            else:
                # Average serve probability so far
                mask_a = (server_sequence[:i+1] == 0)
                mask_b = (server_sequence[:i+1] == 1)
                
                if mask_a.sum() > 0:
                    p_a = point_win_probs_a[:i+1][mask_a].mean()
                else:
                    p_a = 0.65  # Default
                
                if mask_b.sum() > 0:
                    p_b = point_win_probs_b[:i+1][mask_b].mean()
                else:
                    p_b = 0.65  # Default
            
            match_win_probs[i] = self.prob_win_match_a(p_a, p_b)
        
        return match_win_probs


def test_calculator():
    """Test the match probability calculator."""
    calc = MatchProbabilityCalculator(best_of_five=True)
    
    # Test with typical serve probabilities
    p_a = 0.65  # Player A wins 65% of points on serve
    p_b = 0.62  # Player B wins 62% of points on serve
    
    print("=" * 50)
    print("Match Probability Calculator Test")
    print("=" * 50)
    print(f"\nPlayer A serve win probability: {p_a:.1%}")
    print(f"Player B serve win probability: {p_b:.1%}")
    
    hold_a = calc.hold_serve_prob(p_a)
    hold_b = calc.hold_serve_prob(p_b)
    print(f"\nPlayer A hold probability: {hold_a:.1%}")
    print(f"Player B hold probability: {hold_b:.1%}")
    
    set_prob = calc.prob_win_set_a(p_a, p_b)
    print(f"\nPlayer A set win probability: {set_prob:.1%}")
    
    match_prob = calc.prob_win_match_a(p_a, p_b)
    print(f"Player A match win probability: {match_prob:.1%}")
    
    return calc


if __name__ == "__main__":
    test_calculator()
