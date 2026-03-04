import pytest
import math
from collections import OrderedDict

from solution_task1 import ShannonFanoEliasBinaryCoder


class TestShannonFanoEliasCoder:
    """Тесты для класса ShannonFanoEliasBinaryCoder"""

    def test_init_valid_probabilities(self):
        "Тест инициализации"
        probs = OrderedDict([
            ('a', 0.5),
            ('b', 0.25),
            ('c', 0.25)
        ])
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert coder is not None
        assert coder._probabilities == probs

    def test_init_invalid_sum(self):
        """Тест 2: Ошибка при суммировании вероятностей"""
        probs = OrderedDict([
            ('a', 0.5),
            ('b', 0.3),
            ('c', 0.3)
        ])
        with pytest.raises(ValueError, match="Сумма вероятностей не равна 1"):
            ShannonFanoEliasBinaryCoder(probs)
    def test_init_negative_probability(self):
        """Тест 3: Ошибка при отрицательной вероятности"""
        probs = OrderedDict([
            ('a', 0.5),
            ('b', -0.1),
            ('c', 0.6)
        ])
        with pytest.raises(ValueError, match="Вероятности должны быть неотрицательны"):
            ShannonFanoEliasBinaryCoder(probs)
    
    def test_entropy_calculation(self):
        """Тест 4: Вычисление энтропии"""
        probs = OrderedDict([
            ('a', 0.5),
            ('b', 0.25),
            ('c', 0.25)
        ])
        coder = ShannonFanoEliasBinaryCoder(probs)
        
        
        expected = - (0.5 * math.log2(0.5) + 
                    0.25 * math.log2(0.25) + 
                    0.25 * math.log2(0.25))
        
        assert math.isclose(coder.entropy, expected, rel_tol=1e-9)

    def test_code_lengths(self):
        """Тест 5: Проверка, что у всех символов есть коды"""
        probs = OrderedDict([
            ('a', 0.75),
            ('b', 0.025),
            ('c', 0.1),
            ('d', 0.05),
            ('e', 0.075)
        ])
        coder = ShannonFanoEliasBinaryCoder(probs)
        
       
        assert coder._code_table is not None
        
        
        for symbol in probs.keys():
            assert symbol in coder._code_table
            assert len(coder._code_table[symbol]) > 0
    
    def test_expected_code_length(self):
        """Тест 6: Проверка вычисления средней длины."""
        probs = OrderedDict([
            ('a', 0.5),
            ('b', 0.25),
            ('c', 0.25)
        ])
        coder = ShannonFanoEliasBinaryCoder(probs)
        
        
        assert coder.expected_code_length >= coder.entropy
        
        
        assert coder.expected_code_length <= 3.0

    def test_coding_efficiency(self):
        """Тест 7: Проверка эффективности кодирования."""
        probs = OrderedDict([
            ('a', 0.5),
            ('b', 0.25),
            ('c', 0.25)
        ])
        coder = ShannonFanoEliasBinaryCoder(probs)
        
        
        efficiency = coder.coding_efficiency
        assert 0 <= efficiency <= 1
        
        
    

    def test_short_and_long_code(self):
        """Тест 8: Короткий и длинный код"""
        
        probs = OrderedDict([
            ('a', 0.999),
            ('b', 0.001)
        ])
        coder = ShannonFanoEliasBinaryCoder(probs)
        assert len(coder._code_table['a']) == 1  
        assert len(coder._code_table['b']) > 5  