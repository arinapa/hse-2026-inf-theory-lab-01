import math
from codinglab import PrefixCodeTree, PrefixEncoderDecoder

# typing:
from codinglab import SourceChar, ChannelChar
from typing import Sequence, Dict


"""
Shannon encoder-decoder implementation for the coding experiments library.

This module implements Shannon coding, a prefix coding algorithm that
assigns codes based on cumulative probabilities. While not optimal,
it provides a simple implementation that guarantees codes of length
ceil(-log2(p)) for symbols with probability p.
"""


def qrepr(p: float, q: int, prec: int) -> list[str]:
    number = int(p * q**prec)
    result = [""] * prec
    for digit in range(prec):
        result[digit] = str((number // (q**digit)) % q)
    return list(reversed(result))


class ShannonEncoder(PrefixEncoderDecoder[SourceChar, ChannelChar]):
    """
    Shannon encoder-decoder for prefix codes.

    This encoder implements the Shannon coding algorithm, which
    constructs prefix codes based on cumulative probabilities of
    symbols sorted in decreasing order of probability. The code
    length for a symbol with probability p is ceil(-log2(p)).

    Attributes:
        _probabilities: Dictionary mapping source symbols to their probabilities
    """

    def __init__(
        self,
        probabilities: Dict[SourceChar, float],
        channel_alphabet: Sequence[ChannelChar],
    ) -> None:
        """
        Initialize the Shannon encoder with symbol probabilities.

        Args:
            probabilities: Dictionary mapping source symbols to their
                           probabilities (must sum to 1.0)
            channel_alphabet: Sequence of channel symbols for encoding

        Raises:
            ValueError: If probabilities don't sum to approximately 1.0
        """
        # Validate probabilities sum to 1.0
        prob_sum = sum(probabilities.values())
        if not math.isclose(
            prob_sum, 1.0, rel_tol=1e-9
        ):  # Allow for floating-point errors
            raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")

        self._probabilities = probabilities
        self._base = len(channel_alphabet)

        super().__init__(list(probabilities.keys()), channel_alphabet)

    def _build_prefix_code_tree(self) -> None:
        """Build Shannon prefix code tree."""
        # Sort symbols by decreasing probability
        sorted_symbols = sorted(
            self._probabilities.items(),
            key=lambda x: (-x[1], x[0]),  # Sort by probability desc, then symbol
        )

        # Calculate cumulative probabilities
        cumulative = 0.0
        cumulative_probs = []

        for symbol, prob in sorted_symbols:
            cumulative_probs.append((symbol, prob, cumulative))
            cumulative += prob

        # Build prefix code tree
        self._tree = PrefixCodeTree()

        for symbol, prob, cum_prob in cumulative_probs:
            # Calculate code length: ceil(-log2(p))
            if prob > 0:
                code_length = math.ceil(-math.log2(prob))
            else:
                code_length = 0

            # Convert cumulative probability to binary fraction
            # and take first code_length bits
            code = qrepr(cum_prob, self._base, code_length)

            channel_code = [
                self._channel_alphabet[0] if bit == "0" else self._channel_alphabet[1]
                for bit in code
            ]
            self._tree.insert_code(channel_code, symbol)

        # Build code table from tree
        self._build_table_from_tree()

    @property
    def expected_code_length(self) -> float:
        """
        Calculate the expected code length.

        Returns:
            Expected number of channel symbols per source symbol,
            weighted by symbol probabilities
        """
        if not self._code_table:
            return 0.0

        total = 0.0
        for symbol, prob in self._probabilities.items():
            if symbol in self._code_table:
                total += prob * len(self._code_table[symbol])
        return total

    @property
    def entropy(self) -> float:
        """
        Calculate the Shannon entropy of the source.

        Returns:
            Shannon entropy in bits (for binary channel)
        """
        h = 0.0
        for prob in self._probabilities.values():
            if prob > 0:
                h -= prob * math.log2(prob)
        return h

    @property
    def coding_efficiency(self) -> float:
        """
        Calculate the coding efficiency.

        Returns:
            Ratio of entropy to expected code length,
            representing how close the code is to optimal
        """
        expected_len = self.expected_code_length
        if expected_len == 0:
            return 0.0
        return self.entropy / expected_len
