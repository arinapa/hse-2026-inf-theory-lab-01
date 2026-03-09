import heapq
import math
from codinglab import PrefixCodeTree, TreeNode, PrefixEncoderDecoder
from codinglab import ChannelChar
from typing import Optional, Dict, List, Tuple
from collections import Counter
from dataclasses import dataclass
from enum import Enum


class BinaryAlphabet(str, Enum):
    zero = "0"
    one = "1"


@dataclass
class NGram:
    symbols: Tuple[str, ...]

    def __str__(self) -> str:
        return "".join(self.symbols)

    def __hash__(self) -> int:
        return hash(self.symbols)

    def __eq__(self, other) -> bool:
        return self.symbols == other.symbols


@dataclass()
class NGramHuffmanNode(TreeNode[ChannelChar, NGram]):
    freq: float

    def __lt__(self, other):
        return self.freq < other.freq


class NGramHuffmanEncoder(PrefixEncoderDecoder[NGram, BinaryAlphabet]):
    """
    Huffman encoder-decoder for optimal prefix codes for N-gramms.

    This encoder implements the Huffman coding algorithm, which breaks the text into blocks of length n and creates an
    optimal prefix code for a given set of block frequencies. Shorter codes are used for more
    frequent blocks, which minimizes the expected code length.

    Attributes:
        _frequencies: Dictionary mapping source symbols to their frequencies
        n : длина одного блока
        padding_char : символ для дополнения текста до длины кратной n
    """

    def __init__(self, n: int = 1, padding_char: str = "_"):
        self.n = n
        self.padding_char = padding_char
        self._frequencies: Dict[NGram, float] = {}

        self._temp_alphabet = ["dummy"]
        # super().__init__(list(self._frequencies.keys()) if self._frequencies else ['dummy'], [BinaryAlphabet.zero, BinaryAlphabet.one])

    def build_from_text(self, text: str) -> None:
        padded_text = self._pad_text(text)
        ngrams = self._split_into_ngrams(padded_text)

        total = len(ngrams)
        counter = Counter(ngrams)
        self._frequencies = {
            NGram(tuple(ngram)): count / total for ngram, count in counter.items()
        }

        self._source_alphabet = list(self._frequencies.keys())
        source_alphabet = list(self._frequencies.keys())
        super().__init__(source_alphabet, [BinaryAlphabet.zero, BinaryAlphabet.one])
        self._build_prefix_code_tree()

    def _pad_text(self, text: str) -> str:
        """Дополняем текст до длины кратной n"""
        remainder = len(text) % self.n
        if remainder == 0:
            return text
        padding_len = self.n - remainder
        return text + self.padding_char * padding_len

    def _split_into_ngrams(self, text: str) -> List[str]:
        """Разбиваем текст на блоки длины n"""
        return [text[i : i + self.n] for i in range(0, len(text), self.n)]

    def encode_text(self, text: str) -> List[BinaryAlphabet]:
        """Кодируем текст по блокам"""
        if not self._code_table:
            raise ValueError("Нет таблицы кодов")

        padded_text = self._pad_text(text)
        ngrams = self._split_into_ngrams(padded_text)

        encoded: List[BinaryAlphabet] = []
        for ngram in ngrams:
            code = self._code_table[NGram(tuple(ngram))]
            encoded.extend(code)

        return encoded

    def _build_prefix_code_tree(self) -> None:
        """Build Huffman tree using the priority queue algorithm."""
        if not self._frequencies:
            return

        heap: List[NGramHuffmanNode] = []
        for ngram, freq in self._frequencies.items():
            heapq.heappush(heap, NGramHuffmanNode(freq=freq, value=ngram))

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = NGramHuffmanNode(
                freq=left.freq + right.freq,
                value=None,
                children={"0": left, "1": right},
            )
            heapq.heappush(heap, parent)

        root_huffman = heap[0] if heap else None
        self._tree = PrefixCodeTree(root_huffman)
        self._build_table_from_tree()

    def _huffman_to_prefix_tree(
        self, huffman_node: Optional[NGramHuffmanNode]
    ) -> PrefixCodeTree[BinaryAlphabet, NGram]:
        """Convert Huffman tree to prefix code tree."""
        prefix_tree: PrefixCodeTree = PrefixCodeTree()

        def build_tree(
            current_huffman: NGramHuffmanNode, current_prefix: List[BinaryAlphabet]
        ) -> None:
            if current_huffman.value is not None:
                prefix_tree.insert_code(current_prefix, current_huffman.value)
            else:
                left_child = current_huffman.children.get("0")
                right_child = current_huffman.children.get("1")

                if left_child is not None and isinstance(left_child, NGramHuffmanNode):
                    build_tree(
                        left_child,
                        current_prefix + [BinaryAlphabet.zero],
                    )
                if right_child is not None and isinstance(
                    right_child, NGramHuffmanNode
                ):
                    build_tree(
                        right_child,
                        current_prefix + [BinaryAlphabet.one],
                    )

        if huffman_node:
            build_tree(huffman_node, [])

        return prefix_tree

    @property
    def expected_code_length_per_block(self) -> float:
        """
        Calculate the expected code length.

        Returns:
            Expected number of channel symbols per source symbol,
            weighted by symbol frequencies
        """
        if not self._code_table:
            return 0.0

        total = 0.0
        for ngram, freq in self._frequencies.items():
            if ngram in self._code_table:
                total += freq * len(self._code_table[ngram])
        return total

    @property
    def expected_code_length_per_symbol(self) -> float:
        return self.expected_code_length_per_block / self.n

    @property
    def entropy_per_block(self) -> float:
        h = 0.0
        for freq in self._frequencies.values():
            if freq > 0:
                h -= freq * math.log2(freq)
        return h

    @property
    def entropy_per_symbol(self) -> float:
        """
        Calculate the Shannon entropy of the source.

        Returns:
            Shannon entropy in bits (for binary channel)
        """
        return self.entropy_per_block / self.n

    @property
    def coding_efficiency(self) -> float:
        """
        Calculate the coding efficiency.

        Returns:
            Ratio of entropy to expected code length,
            representing how close the code is to optimal
        """

        expected = self.expected_code_length_per_block
        if expected == 0:
            return 0.0
        return self.entropy_per_block / expected
