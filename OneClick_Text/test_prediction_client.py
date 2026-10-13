"""The retained prediction client returns model data without clock filtering."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from OneClick_Text.prediction_client import CachedOneClickWordClient, perfect_observation


class PredictionClientTests(unittest.TestCase):
    def test_request_and_disk_cache_preserve_raw_predictions(self):
        payload = {'prefix': [{'text': 'hello', 'logProb': -1.0}] * 2,
                   'best': [{'text': 'hello world', 'logProb': -2.0}]}
        observations = [perfect_observation('h')]
        response = Mock()
        response.json.return_value = payload
        with tempfile.TemporaryDirectory() as tmp, patch(
            'OneClick_Text.prediction_client.requests.post', return_value=response
        ) as post:
            client = CachedOneClickWordClient(Path(tmp), word_api_url='https://example.invalid/words')
            expected = (payload['prefix'], payload['best'])
            self.assertEqual(client.get_word_predictions('left ', observations), expected)
            body = post.call_args.kwargs['json']
            self.assertEqual(body, {'left': 'left ', 'numBest': 10, 'numPrefix': 25,
                                    'distribs': observations, 'config': 'nomon'})
            cached = CachedOneClickWordClient(Path(tmp), word_api_url=client.word_api_url)
            self.assertEqual(cached.get_word_predictions('left ', observations), expected)
            self.assertEqual(post.call_count, 1)
            self.assertEqual((client.request_count, cached.cache_hit_count), (1, 1))
            cached.get_word_predictions('different ', observations)
            self.assertEqual(post.call_count, 2)

    def test_invalid_response_is_reported(self):
        response = Mock()
        response.json.return_value = {'prefix': {}, 'best': []}
        with tempfile.TemporaryDirectory() as tmp, patch(
            'OneClick_Text.prediction_client.requests.post', return_value=response
        ):
            with self.assertRaisesRegex(RuntimeError, 'prefix and best lists'):
                CachedOneClickWordClient(Path(tmp)).get_word_predictions('', [])

    def test_observation_preserves_prediction_alphabet(self):
        distribution = perfect_observation("'")['distrib']
        self.assertEqual([x['text'] for x in distribution], list("abcdefghijklmnopqrstuvwxyz'"))
        self.assertAlmostEqual(sum(math.exp(x['logProb']) for x in distribution), 1.0)
        self.assertAlmostEqual(math.exp(distribution[-1]['logProb']), .99)


if __name__ == '__main__':
    unittest.main()
