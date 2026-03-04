from codinglab import PrefixEncoderDecoder, SourceChar
from collections import OrderedDict
from enum import Enum
import math

class BinaryAlphabet(str, Enum):
    zero = "0"
    one = "1"



EPS = 1e-9

class ShannonFanoEliasBinaryCoder(PrefixEncoderDecoder[SourceChar, BinaryAlphabet]):
    """
    Shannon-Fano-Elias Coder for prefix codes.

    This coder implements the Shannon-Fano-Elias coding algorithm, which
    constructs prefix codes based on the cumulative distribution function.
    For each symbol x with probability p(x), the code length is:
    l(x) = -⌈log₂(p(x))⌉ + 1
    The code is the binary representation of the modified cumulative
    probability F̄(x) = Σ_{a<x} p(a) + p(x)/2, truncated to l(x) bits.

    Attributes:
        _probabilities: Ordered dictionary mapping source symbols to their probabilities
        _cumulative_probs: Dictionary mapping symbols to their cumulative probabilities
        _modified_cumulative: Dictionary mapping symbols to their modified cumulative F̄(x)
    """

    def __init__(
        self,
        probabilities: OrderedDict[SourceChar, float],
    ) -> None:
        """
        Initialize the Shannon-Fano-Elias encoder with symbol probabilities.

        Args:
            probabilities: Ordered dictionary mapping source symbols to their
                           probabilities (must sum to 1.0). The order determines
                           the cumulative probability calculation.

        Raises:
            ValueError: If probabilities don't sum to approximately 1.0
                       or if any probability is non-positive
        """
        sum_prob = sum(probabilities.values())
        if abs(sum_prob - 1.0) > EPS: 
            raise ValueError("Сумма вероятностей не равна 1")
        
        for prob in probabilities.values():
            if prob <= 0:
                raise ValueError('Одна из вероятностей отрицательна')

        self._probabilities = probabilities.copy()

        self._comulative_probs = OrderedDict()
        comulative = 0.0
        for symbol, prob in self._probabilities.items():
            self._comulative_probs[symbol] = comulative
            comulative += prob

        self._modified_cumulative = OrderedDict()
        for symbol, prob in self._probabilities.items():
            self._modified_cumulative[symbol] = self._comulative_probs[symbol] + prob/2.0

        self._build_prefix_code_tree()


    def _build_prefix_code_tree(self) -> None:
        codes =  {}

        for symbol, prob in self._probabilities.items():
            code_len = - math.ceil(math.log2(prob)) + 1

        

            code = ""
            current = self._modified_cumulative[symbol]
            
            for _ in range(code_len):
                current *= 2
                bit = int(current)
                code += str(bit)
                current -= bit

            codes[symbol] = [BinaryAlphabet.zero if bit == '0' else BinaryAlphabet.one for bit in code]
        self._code_table = codes

    @property
    def expected_code_length(self) -> float:
        """
        Calculate the expected code length.

        Returns:
            Expected number of channel symbols per source symbol,
            weighted by symbol probabilities
        """
        expected_len = 0.0

        for symbol, prob in self._probabilities.items():
            expected_len += prob * len (self.code_table[symbol])
        
        return expected_len
    

    @property
    def entropy(self) -> float:
        """
        Calculate the Shannon entropy of the source.

        Returns:
            Shannon entropy in bits (for binary channel)
        """
        
        entropy = 0.0

        for prob in self._probabilities.values():
            entropy -= prob * math.log2(prob)
        return entropy

    @property
    def coding_efficiency(self) -> float:
        """
        Calculate the coding efficiency.

        Returns:
            Ratio of entropy to expected code length,
            representing how close the code is to optimal
        """
        return self.entropy / self.expected_code_length