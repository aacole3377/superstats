import pytest
from superstats import get_std

# Test a valid array
def test_get_std_valid():
    result = get_std([2, 4, 6, 8])
    expected = 2.58198889747  # you can use round to make this cleaner
    assert round(result, 2) == round(expected, 2)

# Test with empty list
def test_get_std_empty():
    with pytest.raises(ValueError):
        get_std([])

# Test with invalid data type
def test_get_std_bad_type():
    with pytest.raises(ValueError):
        get_std([1, 2, 'a', 4])

@pytest.mark.parametrize("data, expected", [
    ([2, 4, 6, 8], 2.58),
    ([10, 20, 30], 10.0),
    ([5, 5, 5, 5], 0.0)
])
def test_get_std_param(data, expected):
    assert round(get_std(data), 2) == expected
