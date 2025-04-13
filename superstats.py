import pandas as pd
import numpy as np
from typing import Union, Sequence
from dataclasses import dataclass
from scipy.stats import t

Number = Union[int, float]
NumericSequence = Sequence[Number]

def get_variance(arr: NumericSequence) -> float:
    if len(arr) == 0:
        return "Empty data"
    arr = np.array(arr)
    arr_mean = np.sum(arr) / len(arr)
    
    distance = [i-arr_mean for i in arr]
    distance_squared = [i**2 for i in distance]
    distance_sum = np.sum(distance_squared)

    variance = distance_sum/(len(distance) - 1)
    return variance

def get_std(arr: NumericSequence) -> float:
    if len(arr) == 0:
        return "Empty data"
    variance = get_variance(arr)
    std = np.sqrt(variance)
    return float(std)

def get_covariance(x_values: NumericSequence, y_values: NumericSequence) -> float:
    if len(x_values) != len(y_values):
        return 
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    products = [(x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)]
    
    cov = sum(products) / (len(products)-1)
    
    return float(cov)

def get_t_score(x_values : NumericSequence, y_values : NumericSequence) -> float:

    if len(x_values) != len(y_values):
        return "Invalid array sizes"

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

def get_p_value(t_score: float, df: float, two_tailed: bool = True) -> float:
    """
    Returns the p-value for a given t-score and degrees of freedom.
    """
    if two_tailed:
        return 2 * t.sf(abs(t_score), df)  
    else:
        return t.sf(abs(t_score), df)      

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
