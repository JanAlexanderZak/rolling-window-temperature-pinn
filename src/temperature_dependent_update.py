import time

import numpy as np
import plotly.graph_objects as go

from scipy.interpolate import interp1d

np.random.seed(6020)


def lookup_table_method(
    temp_samples: np.ndarray,
    material_param_array: np.ndarray,
) -> np.ndarray:
    """ Lookup table approach
    """
    temp_samples_rounded = np.round(temp_samples).astype(int)
    material_param_values = material_param_array[temp_samples_rounded]
    return material_param_values

def numpy_interp_method(
    temp_samples: np.ndarray,
    temp_measured: np.ndarray,
    material_measured: np.ndarray,
) -> np.ndarray:
    """ NumPy linear interpolation
    """
    return np.interp(temp_samples, temp_measured, material_measured)

def scipy_cubic_method(
    temp_samples: np.ndarray,
    temp_measured: np.ndarray,
    material_measured: np.ndarray,
) -> np.ndarray:
    """ SciPy cubic interpolation
    """
    f = interp1d(temp_measured, material_measured, kind='cubic')
    return f(temp_samples)

def benchmark(
    method_func,
    *args,
    n_runs=10000,
):
    """ Benchmark a method with multiple runs for accurate timing
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = method_func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


def compare_efficiency():
    # Create test data
    measured_material_parameters = 50 + 0.1 * np.arange(1001)
    measured_temperature_range = np.arange(0, 1001, 1)
    sampled_temperatures = np.random.uniform(0, 1000, 10000)
    
    # * 1. Lookup Table
    lookup_time, lookup_std = benchmark(
        lookup_table_method,
        sampled_temperatures,
        measured_material_parameters,
    )
    
    # * 2. NumPy Interpolation
    numpy_time, numpy_std = benchmark(
        numpy_interp_method,
        sampled_temperatures,
        measured_temperature_range,
        measured_material_parameters,
    )
    
    # * 3. SciPy Cubic Interpolation
    scipy_time, scipy_std = benchmark(
        scipy_cubic_method,
        sampled_temperatures,
        measured_temperature_range,
        measured_material_parameters,
    )

    return lookup_time, lookup_std, numpy_time, numpy_std, scipy_time, scipy_std
