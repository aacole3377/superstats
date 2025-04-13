# SuperStats Documentation

SuperStats is a lightweight Python statistics utility library designed to help users perform basic statistical calculations, linear regression, correlation, and t-tests with clear, interpretable functions.

## Installation
Clone or download the repository from GitHub:

```bash
git clone https://github.com/yourusername/superstats.git
```

## Usage
Import the functions into your Python project:

```python
from superstats import (
    get_variance,
    get_std,
    get_covariance,
    get_t_score,
    get_p_value,
    get_correlation,
    linear_regression,
    LinearModel
)
```

---

## Functions

### `get_variance(arr)`
**Description:** Calculates the sample variance of a numeric sequence.
- **Input:** `arr` — sequence of numbers (list, tuple, or NumPy array)
- **Returns:** Sample variance (float)

### `get_std(arr)`
**Description:** Calculates the sample standard deviation.
- **Input:** `arr` — sequence of numbers
- **Returns:** Standard deviation (float)

### `get_covariance(x_values, y_values)`
**Description:** Calculates the sample covariance between two numeric sequences.
- **Inputs:** `x_values`, `y_values` — sequences of equal length
- **Returns:** Covariance (float)

### `get_t_score(x_values, y_values)`
**Description:** Computes the t-score for two independent samples.
- **Inputs:** Two equal-length numeric sequences
- **Returns:** t-score (float)

### `get_p_value(t_score, df, two_tailed=True)`
**Description:** Computes the p-value associated with a t-score and degrees of freedom.
- **Inputs:**
  - `t_score`: t-statistic
  - `df`: degrees of freedom
  - `two_tailed`: Boolean (default True)
- **Returns:** p-value (float)

### `get_correlation(x_values, y_values)`
**Description:** Computes the Pearson correlation coefficient between two numeric sequences.
- **Inputs:** Two numeric sequences
- **Returns:** Correlation coefficient (float)

### `linear_regression(x_values, y_values)`
**Description:** Computes the line of best fit using simple linear regression.
- **Inputs:**
  - `x_values`, `y_values`: numeric sequences
- **Returns:** `LinearModel` object

---

## Class: `LinearModel`
Represents a fitted linear regression model.

### Attributes:
- `slope`: float
- `intercept`: float

### Method:
#### `predict(x)`
- **Input:** Single number or list of numbers
- **Returns:** Predicted value(s) using the regression model

---

## Example
```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

model = linear_regression(x, y)
print("Slope:", model.slope)
print("Intercept:", model.intercept)
print("Prediction for x=6:", model.predict(6))
```

---

## Notes
- All calculations are done using NumPy for efficiency.
- Missing values and zero-length sequences are not handled beyond basic checks.

---

## License
MIT License

## Author
Aaron Cole
