"""Module for black scholes tests"""

from opcoes.black_scholes import bs_d_one


class TestBsDOne:
    """tests for function bs_d_one"""

    def test_bs_d_one(self):
        assert bs_d_one(100, 100, 0, 0, 16, 1) > 0
