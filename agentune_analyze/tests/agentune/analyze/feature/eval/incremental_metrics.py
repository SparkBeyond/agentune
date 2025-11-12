from abc import ABC, abstractmethod
from concurrent.futures import Future, ProcessPoolExecutor
from queue import Queue
from typing import cast, override

import numpy as np
import polars as pl
from attr import frozen
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def generate_regression_data(size: int, correlation: float, random_state: int) -> pl.DataFrame:
    # Nondefault generator for back compat
    rng = np.random.Generator(np.random.MT19937(random_state))
    feature = rng.normal(0, 1, size)
    noise = rng.normal(0, 1, size)
    target = correlation * feature + np.sqrt(1 - correlation**2) * noise
    return pl.DataFrame({
        'feature': feature,
        'target': target
    })

def generate_classification_data(size: int, correlation_strength: float, n_classes: int, random_state: int) -> pl.DataFrame:
    """Generate classification data with controlled correlation between feature and target"""
    rng = np.random.default_rng(random_state)
    
    # Generate feature
    feature = rng.normal(0, 1, size)
    
    # Create class boundaries based on feature quantiles for strong correlation
    if correlation_strength > 0:
        # Sort feature to create class boundaries
        sorted_indices = np.argsort(feature)
        class_size = size // n_classes
        
        target = np.zeros(size, dtype=int)
        for i in range(n_classes):
            start_idx = i * class_size
            end_idx = (i + 1) * class_size if i < n_classes - 1 else size
            target[sorted_indices[start_idx:end_idx]] = i
        
        # Add noise to reduce correlation
        noise_ratio = 1 - correlation_strength
        n_noise = int(size * noise_ratio)
        noise_indices = rng.choice(size, n_noise, replace=False)
        target[noise_indices] = rng.integers(0, n_classes, n_noise)
    else:
        # Random assignment
        target = rng.integers(0, n_classes, size)
    
    return pl.DataFrame({
        'feature': feature,
        'target': target
    })

def calculate_differential_entropy(data: np.ndarray) -> float:
    """Calculate differential entropy of continuous data.
    For normally distributed data: H(X) = 0.5 * log(2πeσ²)
    """
    # Estimate variance
    variance = np.var(data, ddof=1)  # Use sample variance
    
    # Differential entropy for normal distribution
    entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
    
    return entropy

def regression_rig(feature: np.ndarray, target: np.ndarray) -> float:
    # Calculate mutual information between feature and target
    mi = mutual_info_regression(feature.reshape(-1, 1), target, random_state=42, discrete_features=False)[0]
    
    # Calculate differential entropy of target directly
    h_target = calculate_differential_entropy(target)
    
    return mi / h_target

def calculate_shannon_entropy(target: np.ndarray) -> float:
    """Calculate Shannon entropy for discrete/categorical data.
    H(Y) = -Σ p(y) * log(p(y))
    """
    # Get unique values and their counts
    unique_values, counts = np.unique(target, return_counts=True)
    
    # Calculate probabilities
    probabilities = counts / len(target)
    
    # Calculate Shannon entropy (using natural log to match MI calculations)
    entropy = -np.sum(probabilities * np.log(probabilities))
    
    return entropy

def classification_rig(feature: np.ndarray, target: np.ndarray) -> float:
    # Calculate mutual information between feature and target
    mi = mutual_info_classif(feature.reshape(-1, 1), target)[0]
    
    # Calculate Shannon entropy of target directly
    h_target = calculate_shannon_entropy(target)
    
    return mi / h_target

class Metric(ABC):
    @abstractmethod
    def calculate(self, correlation: float, total_size: int, sample_sizes: list[int],
                  random_seed: int) -> tuple[float, pl.DataFrame]: 
        """Return the 'true' metric value calculated on the total_size rows, and a dataframe
        with columns 'size', 'metric', 'error' where 'error' is the relative error of the metric
        on the sample of size 'size' compared to the 'true' metric value.
        """
        ...
    
@frozen
class ClassificationRig(Metric):
    n_classes: int
    
    @override
    def calculate(self, correlation: float, total_size: int, sample_sizes: list[int],
                  random_seed: int) -> tuple[float, pl.DataFrame]:
        all_data = generate_classification_data(total_size, correlation, self.n_classes, random_seed)
        true_rig = classification_rig(all_data['feature'].to_numpy(), all_data['target'].to_numpy())
        #print(f'true rig: {true_rig} for correlation {correlation}')
        sample_rigs = [classification_rig(all_data['feature'].to_numpy()[:size], all_data['target'].to_numpy()[:size]) for size in sample_sizes]
        errors = [abs((sample_rig - true_rig) / true_rig) for sample_rig in sample_rigs]
        return true_rig, pl.DataFrame({
            'size': sample_sizes,
            'rig': sample_rigs,
            'error': errors
        })

@frozen
class ClassificationCorr(Metric):
    n_classes: int

    @override
    def calculate(self, correlation: float, total_size: int, sample_sizes: list[int],
                  random_seed: int) -> tuple[float, pl.DataFrame]:
        all_data = generate_classification_data(total_size, correlation, self.n_classes, random_seed)
        true_corr = np.corrcoef(all_data['feature'].to_numpy(), all_data['target'].to_numpy(), rowvar=False)[0, 1]
        #print(f'true corr: {true_corr} for correlation {correlation}')
        sample_corrs = [np.corrcoef(all_data['feature'].to_numpy()[:size], all_data['target'].to_numpy()[:size], rowvar=False)[0, 1] for size in sample_sizes]
        errors = [abs((sample_corr - true_corr) / true_corr) for sample_corr in sample_corrs]
        return true_corr, pl.DataFrame({
            'size': sample_sizes,
            'corr': sample_corrs,
            'error': errors
        })

class RegressionCorr(Metric):
    @override
    def calculate(self, correlation: float, total_size: int, sample_sizes: list[int],
                  random_seed: int) -> tuple[float, pl.DataFrame]:
        all_data = generate_regression_data(total_size, correlation, random_seed)
        true_corr, true_p = spearmanr(all_data['feature'].to_numpy(), all_data['target'].to_numpy())
        #print(f'true corr: {true_corr}, true_p: {true_p} for correlation {correlation}')
        sample_corrs = [spearmanr(all_data['feature'].head(size).to_numpy(), all_data['target'].head(size).to_numpy())[0] for size in sample_sizes]
        errors = [abs((sample_corr - true_corr) / true_corr) for sample_corr in sample_corrs]
        return cast(float, true_corr), pl.DataFrame({
            'size': sample_sizes,
            'corr': sample_corrs,
            'error': errors
        })


def plot() -> None:
    rng = np.random.default_rng(42)
    #for correlation in [0.01, 0.025, 0.05, 0.12, 0.2, 0.3, 0.4, 0.5]:
    for correlation, target_rig in [(0.15, 0.018), (0.2, 0.028), (0.3, 0.063), (0.4, 0.12), (0.5, 0.19), (0.7, 0.39)]:
        plt.figure(figsize=(20, 12))  # Set a large figure size
        plt.axhline(y=0, linestyle='dashed', color='black')
        plt.xlabel('Sample size')
        plt.ylabel('Relative error %')
        true_values = []
        for random_seed in rng.integers(0, 1000000, size=20):
            #true_value, df = regression_corrs(correlation=correlation, total_size=100_000, sample_sizes=sample_sizes, random_seed=random_seed) 
            true_value, df = ClassificationRig(n_classes=2).calculate(correlation=correlation, total_size=100_000, sample_sizes=sample_sizes, random_seed=int(random_seed))
            true_values.append(true_value)
            accuracy = abs(true_value / target_rig)
            if accuracy > 0.95:
                plt.plot(df['size'], df['error'], label=f'{true_value:.3f} (seed {random_seed})', linestyle='solid')
                #plt.legend()
                plt.xscale('log')
            else:
                print(f'Low accuracy {accuracy} for correlation {correlation} and seed {random_seed}, skipping; '
                      f'got rig={true_value:.3f} but wanted target_rig={target_rig:.3f}')
        plt.title(f'Binary classification RIG estimation error, true RIG ~= {np.mean(true_values):.3f}')
        plt.savefig(f'classifition_rig_{np.mean(true_values):.3f}.png')
        #plt.show()
        plt.close()


def max_error_one(metric: Metric, sample_sizes: list[int], large_size: int, 
                  target_correlation: float, random_seed: int) -> tuple[float, pl.DataFrame]:
    return metric.calculate(target_correlation, large_size, sample_sizes, random_seed)

def max_error(metric: Metric, sample_sizes: list[int], target_correlation: float, 
              large_size: int, samplings: int, true_metric_goal: float, true_metric_max_error: float = 0.05) -> list[float]:
    """For each input sample size, returns the max relative error % (between 0 and 100) of estimating the correlation
    on that many samples. Takes the upper bound (greatest uncertainty) for the different levels of 
    (true) correlation tested.
    """
    result = [0.0 for size in sample_sizes]
    
    queue = Queue[Future[tuple[float, pl.DataFrame]]]()

    with ProcessPoolExecutor() as executor:
        rng = np.random.default_rng(42)
        for _ in range(samplings):
            random_seed = int(rng.integers(0, 1000000))
            future = executor.submit(max_error_one, metric, sample_sizes, large_size, target_correlation, random_seed)
            queue.put(future)

        while not queue.empty():
            future = queue.get()
            true_metric, df = future.result()
            metric_error = abs((true_metric - true_metric_goal) / true_metric_goal)
            if metric_error <= true_metric_max_error:
                for index, (_size, _corr, error) in enumerate(df.iter_rows()):
                    result[index] = max(result[index], error)
                print('.', end='', flush=True)
            else:
                #print(f'\n{metric_error=} {true_metric=} {true_metric_goal=}')
                print('!', end='', flush=True)
        print() # Newline after printing dots


    return result


if __name__ == '__main__':
    sample_sizes=list(range(10, 1500, 10)) + list(range(1500, 10_000, 100))# + list(range(10_000, 20_000, 1000))
    
    # correlation = 0.2
    # true_metric_goal = 0.035 
    # metric = ClassificationRig(n_classes=3)

    # correlation = 0.2
    # true_metric_goal = 0.028
    # metric = ClassificationRig(n_classes=2)

    correlation = 0.05
    true_metric_goal = 0.05
    metric = RegressionCorr()

    large_size = 100_000
    samplings = 1000
    #metric = RegressionCorr()
    errors = max_error(metric, sample_sizes, correlation, large_size, samplings, true_metric_goal, true_metric_max_error=0.1)
    df = pl.DataFrame({'size': sample_sizes, 'error': errors})
    print(df)
    df = df.with_columns(
        max_subsequent_error = pl.col('error').cum_max(reverse=True).round()
    )
    df = df.group_by('max_subsequent_error').agg(pl.col('size').min().alias('min_sample_size')) \
           .sort('max_subsequent_error') \
           .filter(pl.col('max_subsequent_error') < 15) # In the interests of brevity, assume error > 15% is not useful
    
    print(df)
    
    # Print the data for recreation
    max_subsequent_errors = df['max_subsequent_error'].to_list()
    min_sample_sizes = df['min_sample_size'].to_list()
    print(f"df = pl.DataFrame({{'max_subsequent_error': {max_subsequent_errors}, 'min_sample_size': {min_sample_sizes}}})")

