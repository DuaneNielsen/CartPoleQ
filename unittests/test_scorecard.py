from unittest import TestCase
from mentality import LossScoreCard

class TestLossScoreCard(TestCase):
    def test_scorecard(self):
        sc = LossScoreCard()
        sc.addScore([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        sc.addScore([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0])