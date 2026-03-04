import heapq
import math
from codinglab import PrefixCodeTree, TreeNode, PrefixEncoderDecoder

# typing:
from codinglab import SourceChar, ChannelChar
from typing import Sequence, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

class BinaryAlphabet(str, Enum):
    zero: "0"
    one: "1"


@dataclass(kw_only=True)
class HuffmanNode(TreeNode[ChannelChar, SourceChar]):
    """Node in the Huffman tree during construction."""

    freq: float

    def __lt__(self, other: "HuffmanNode[ChannelChar, SourceChar]") -> bool:
        return self.freq < other.freq


class BinaryHuffmanEncoder(PrefixEncoderDecoder[SourceChar, BinaryAlphabet]):
    """
    Huffman encoder-decoder for optimal prefix codes.

    This encoder implements the Huffman coding algorithm, which constructs
    an optimal prefix code for a given set of symbol frequencies. More
    frequent symbols get shorter codes, minimizing the expected code length.

    Attributes:
        _frequencies: Dictionary mapping source symbols to their frequencies
    """

    def __init__(self, frequencies: Dict[SourceChar, float]) -> None:
        """
        Initialize the Huffman encoder with symbol frequencies.

        Args:
            frequencies: Dictionary mapping source symbols to their
                        frequencies (or probabilities)

        Raises:
            ValueError: If frequencies don't match source alphabet,
                       or if channel alphabet is not binary
        """
        self._frequencies = frequencies
        """Dictionary mapping source symbols to their frequencies."""

        super().__init__(frequencies.keys(), ["0", "1"])

    def _build_prefix_code_tree(self) -> None:
        """Build Huffman tree using the priority queue algorithm."""
        # Create leaf nodes for all symbols
        heap = []
        for symbol, freq in self._frequencies.items():
            heapq.heappush(heap, HuffmanNode(freq=freq, value=symbol))

        # Build Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(
                freq=left.freq + right.freq,
                value=None,
                children={"0": left, "1": right},
            )
            heapq.heappush(heap, parent)

        # Convert Huffman tree to prefix code tree
        root_huffman = heap[0] if heap else None
        self._tree = PrefixCodeTree(root_huffman)
        self._build_table_from_tree()

    def _huffman_to_prefix_tree(
        self, huffman_node: Optional[HuffmanNode]
    ) -> PrefixCodeTree[BinaryAlphabet, SourceChar]:
        """Convert Huffman tree to prefix code tree."""
        prefix_tree = PrefixCodeTree()

        def build_tree(
            current_huffman: HuffmanNode, current_prefix: List[BinaryAlphabet]
        ) -> None:
            if current_huffman.symbol is not None:
                # Leaf node: insert code
                prefix_tree.insert_code(current_prefix, current_huffman.symbol)
            else:
                # Internal node: traverse left (0) and right (1)
                if current_huffman.left:
                    build_tree(
                        current_huffman.left,
                        current_prefix + [self._channel_alphabet[0]],
                    )
                if current_huffman.right:
                    build_tree(
                        current_huffman.right,
                        current_prefix + [self._channel_alphabet[1]],
                    )

        if huffman_node:
            build_tree(huffman_node, [])

        return prefix_tree

    @property
    def expected_code_length(self) -> float:
        """
        Calculate the expected code length.

        Returns:
            Expected number of channel symbols per source symbol,
            weighted by symbol frequencies
        """
        if not self._code_table:
            return 0.0

        total = 0.0
        for symbol, freq in self._frequencies.items():
            if symbol in self._code_table:
                total += freq * len(self._code_table[symbol])
        return total

    @property
    def entropy(self) -> float:
        """
        Calculate the Shannon entropy of the source.

        Returns:
            Shannon entropy in bits (for binary channel)
        """
        h = 0.0
        for freq in self._frequencies.values():
            if freq > 0:
                h -= freq * math.log2(freq)
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