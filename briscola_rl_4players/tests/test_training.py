import random
import unittest

from briscola.baselines import RandomPolicy
from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy
from briscola.reinforce import RewardConfig, collect_episode, terminal_reward
from briscola.self_play import SelfPlayConfig, SelfPlayTrainer, SnapshotPool


class TrainingTests(unittest.TestCase):
    def test_one_self_play_update_runs(self):
        extractor = BriscolaFeatureExtractor()
        learner = LinearSoftmaxPolicy.initialize(extractor, rng=random.Random(10))
        pool = SnapshotPool(extractor, max_size=5)
        config = SelfPlayConfig(batch_size=5, learning_rate=0.001, snapshot_interval=10)
        trainer = SelfPlayTrainer(learner=learner, pool=pool, config=config, seed=11)
        stats = trainer.train_update()
        self.assertEqual(stats.episodes, 5)
        self.assertEqual(stats.learner_decisions, 50)
        self.assertGreaterEqual(len(pool.snapshots), 1)

    def test_episode_return_matches_terminal_reward_in_terminal_mode(self):
        extractor = BriscolaFeatureExtractor()
        learner = LinearSoftmaxPolicy.initialize(extractor, rng=random.Random(10))
        reward_config = RewardConfig(mode="combined_terminal", lambda_margin=0.2)

        episode = collect_episode(
            learner_policy=learner,
            partner_policy=RandomPolicy(),
            opponent_policy=RandomPolicy(),
            seed=123,
            learner_seat=2,
            reward_config=reward_config,
        )

        expected_return = terminal_reward(
            scores=episode.final_scores,
            learner_team=episode.learner_team,
            reward_config=reward_config,
        )
        self.assertAlmostEqual(episode.episode_return, expected_return)


if __name__ == "__main__":
    unittest.main()
