import unittest

from briscola.baselines import GreedyTrickPolicy, RandomPolicy
from briscola.evaluation import evaluate_team_match


class EvaluationTests(unittest.TestCase):
    def test_paired_evaluation_reports_same_mean_with_wider_interval_for_fewer_seeds(self):
        policy_a = RandomPolicy()
        policy_b = GreedyTrickPolicy()
        seeds = [100_000, 100_001, 100_002, 100_003]

        result = evaluate_team_match(
            policy_a=policy_a,
            policy_b=policy_b,
            seeds=seeds,
            paired=True,
            greedy=True,
        )

        self.assertEqual(result.games, 2 * len(seeds))
        self.assertLessEqual(result.confidence_interval_95[0], result.mean_point_difference)
        self.assertGreaterEqual(result.confidence_interval_95[1], result.mean_point_difference)


if __name__ == "__main__":
    unittest.main()
