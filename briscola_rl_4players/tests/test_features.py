import unittest

from briscola.env import FourPlayerBriscolaEnv
from briscola.features import BriscolaFeatureExtractor


class FeatureTests(unittest.TestCase):
    def test_feature_vector_has_expected_size(self):
        env = FourPlayerBriscolaEnv(seed=5)
        obs = env.observe(env.current_player)
        extractor = BriscolaFeatureExtractor()
        features = extractor.extract(obs, obs.hand[0])
        self.assertEqual(len(features), extractor.size())

    def test_features_are_numeric(self):
        env = FourPlayerBriscolaEnv(seed=6)
        obs = env.observe(env.current_player)
        extractor = BriscolaFeatureExtractor()
        features = extractor.extract(obs, obs.hand[0])
        self.assertTrue(all(isinstance(value, float) for value in features))

    def test_non_trump_has_some_unobserved_stronger_trumps_early(self):
        env = FourPlayerBriscolaEnv(seed=5)
        obs = env.observe(env.current_player)
        extractor = BriscolaFeatureExtractor()
        card = next(candidate for candidate in obs.hand if candidate.suit != obs.trump_suit)

        feature_index = extractor.feature_names.index("stronger_trumps_not_observed")
        features = extractor.extract(obs, card)
        self.assertGreater(features[feature_index], 0.0)


if __name__ == "__main__":
    unittest.main()
