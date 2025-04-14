import pandas as pd
import numpy as np
from typing import Union, Sequence
from dataclasses import dataclass
from scipy.stats import t
from pprint import pprint

Number = Union[int, float]
NumericSequence = Sequence[Number]

def get_variance(arr: NumericSequence) -> float:
    if len(arr) == 0:
        raise ValueError("Empty data")
    arr = np.array(arr)
    arr_mean = np.sum(arr) / len(arr)
    
    distance = [i-arr_mean for i in arr]
    distance_squared = [i**2 for i in distance]
    distance_sum = np.sum(distance_squared)

    variance = distance_sum/(len(distance) - 1)
    return variance

def get_std(arr: NumericSequence) -> float:
    if len(arr) == 0:
        raise ValueError("Empty data")
    variance = get_variance(arr)
    std = np.sqrt(variance)
    return float(std)

def get_covariance(x_values: NumericSequence, y_values: NumericSequence) -> float:
    if len(x_values) != len(y_values):
        raise ValueError("Data must be of same length") 
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    products = [(x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)]
    
    cov = sum(products) / (len(products)-1)
    
    return float(cov)

def get_two_sample_t_score(x_values : NumericSequence, y_values : NumericSequence) -> float:

    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)

    x_variance = get_variance(x_values)
    y_variance = get_variance(y_values)

    numerator = x_mean - y_mean
    denominator = np.sqrt((x_variance/len(x_values)) + (y_variance/len(y_values)))

    if denominator == 0:
        return None
    
    t_score = numerator / denominator

    return t_score

def two_sample_t_test(
    sample_one: NumericSequence,
    sample_two: NumericSequence,
    two_tailed: bool = True
) -> dict:
    x = np.array(sample_one, dtype=float)
    y = np.array(sample_two, dtype=float)

    n_x, n_y = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = get_variance(x), get_variance(y)

    # 1. Calculate t-score
    numerator = mean_x - mean_y
    denominator = np.sqrt((var_x / n_x) + (var_y / n_y))

    if denominator == 0:
        raise ValueError("Standard error is zero — cannot compute t-score.")

    t_score = numerator / denominator

    # 2. Welch–Satterthwaite approximation for degrees of freedom
    df_numerator = (var_x / n_x + var_y / n_y) ** 2
    df_denominator = ((var_x / n_x) ** 2) / (n_x - 1) + ((var_y / n_y) ** 2) / (n_y - 1)
    degrees_of_freedom = df_numerator / df_denominator

    # 3. Calculate p-value
    if two_tailed:
        p_value = 2 * t.sf(abs(t_score), df=degrees_of_freedom)
    else:
        p_value = t.sf(abs(t_score), df=degrees_of_freedom)

    return {
        "t_score": t_score,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "sample_one_mean": mean_x,
        "sample_two_mean": mean_y,
        "two_tailed": two_tailed
    }


def one_sample_t_test(
    sample: NumericSequence,
    population_mean: float,
    two_tailed: bool = True
) -> dict:
    """
    Performs a one-sample t-test.
    
    Parameters:
        sample (list or array): Sample data
        population_mean (float): The population mean to test against
        two_tailed (bool): Whether to calculate a two-tailed p-value

    Returns:
        dict: {
            't_score': float,
            'degrees_of_freedom': int,
            'p_value': float
        }
    """
    sample = np.array(sample, dtype=float)
    n = len(sample)

    if n < 2:
        raise ValueError("Sample size must be at least 2.")

    sample_mean = np.mean(sample)
    sample_std = get_std(sample)

    # t-statistic formula
    standard_error = sample_std / np.sqrt(n)
    t_score = (sample_mean - population_mean) / standard_error
    df = n - 1

    # p-value from t-distribution
    if two_tailed:
        p_value = 2 * t.sf(abs(t_score), df)
    else:
        p_value = t.sf(abs(t_score), df)

    return {
        "t_score": t_score,
        "degrees_of_freedom": df,
        "p_value": p_value,
        "sample_mean": sample_mean,
        "population_mean": population_mean,
        "two_tailed": two_tailed
    }

def get_correlation(x_values: NumericSequence, y_values: NumericSequence) -> float:
    correlation = get_covariance(x_values, y_values) / (get_std(x_values) * get_std(y_values))
    return correlation

@dataclass
class LinearModel:
    slope: float
    intercept: float

    def predict(self, x: Union[float, Sequence[float]]) -> Union[float, list[float]]:
        if isinstance(x, (int, float)):
            return self.slope * x + self.intercept
        return [self.slope * xi + self.intercept for xi in x]

def linear_regression(x_values: NumericSequence, y_values: NumericSequence) -> tuple: 
    x = np.array(x_values)
    y = np.array(y_values)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    rise = np.sum((x - x_mean) * (y - y_mean))
    run = np.sum((x - x_mean)**2)

    if run == 0:
        return None
    
    slope = rise / run
    intercept = y_mean - (slope * x_mean)

    return LinearModel(slope, intercept)