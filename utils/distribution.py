import json
import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt

def is_normal_distribution(a, alpha=0.05, plot=False):
    """
    Check if array data follows a normal distribution.
    
    Parameters:
    -----------
    a : array-like
        The array to test for normality
    alpha : float, optional (default=0.05)
        The significance level for the test
    plot : bool, optional (default=False)
        Whether to generate a QQ plot for visual inspection
        
    Returns:
    --------
    is_normal : bool
        True if the data likely follows a normal distribution
    p_value : float
        The p-value of the Shapiro-Wilk test
    """
    if len(a) < 3:
        raise ValueError("Array must have at least 3 elements for normality test")
        
    # Shapiro-Wilk test (one of the most powerful normality tests)
    statistic, p_value = stats.shapiro(a)
    is_normal = p_value > alpha
    
    if plot:
        # Create a QQ plot for visual inspection
        plt.figure(figsize=(10, 6))
        
        # QQ plot
        plt.subplot(1, 2, 1)
        stats.probplot(a, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        
        # Histogram with normal curve
        plt.subplot(1, 2, 2)
        plt.hist(a, bins='auto', density=True, alpha=0.7)
        
        # Add a fitted normal distribution line
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        mean, std = np.mean(a), np.std(a)
        p = stats.norm.pdf(x, mean, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title("Histogram with Normal Curve")
        
        plt.tight_layout()
        plt.show()
    
    return is_normal, p_value

# Example usage
if __name__ == "__main__":
    # Generate a normally distributed sample
    normal_data = np.random.normal(size=1000)
    
    # Generate a non-normal sample (e.g., exponential distribution)
    non_normal_data = np.random.exponential(size=1000)
    
    # Test both datasets
    normal_result, normal_p = is_normal_distribution(normal_data, plot=True)
    non_normal_result, non_normal_p = is_normal_distribution(non_normal_data, plot=True)
    
    print(f"Normal data test: Is normal? {normal_result}, p-value: {normal_p:.6f}")
    print(f"Non-normal data test: Is normal? {non_normal_result}, p-value: {non_normal_p:.6f}")
    
    path = "/home/keruihuang/cot_compression/outputs/DeepSeek-R1-Distill-Qwen-7B/sft_1024_3/human_eval/7b/test/samples/predictions.jsonl"
    
    data = []
    with open(path, 'r') as f:
        data = []
        for line in f:
            j = json.loads(line)
            data.append(j['cot_length'])
    
    print(is_normal_distribution(data, plot=True))
            