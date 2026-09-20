"""
Statistics for AI Healthcare Assistant.

This module provides statistical methods used in:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Machine Learning
- Healthcare risk analysis
- Model interpretation

Main topics:
1. Descriptive statistics
2. Variance and standard deviation
3. Quartiles and IQR
4. Covariance and correlation
5. Z-score and standardization
6. Confidence interval
7. Healthcare prevalence
8. Group statistics
"""

from collections import Counter
from math import sqrt


# ============================================================
# 1. BASIC DESCRIPTIVE STATISTICS
# ============================================================

def mean(data):
    """
    Calculate arithmetic mean.

    Formula:
        mean = sum(x_i) / n

    Parameters
    ----------
    data : list
        Numerical data.

    Returns
    -------
    float
        Mean value.
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    return sum(data) / len(data)


def median(data):
    """
    Calculate median.

    Median is the middle value after sorting the data.
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    sorted_data = sorted(data)
    n = len(sorted_data)

    middle = n // 2

    if n % 2 == 0:
        return (sorted_data[middle - 1] + sorted_data[middle]) / 2

    return sorted_data[middle]


def mode(data):
    """
    Calculate mode.

    Returns the value with the highest frequency.

    If multiple values have the same maximum frequency,
    all of them are returned.
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    frequency = Counter(data)
    max_frequency = max(frequency.values())

    modes = [
        value
        for value, count in frequency.items()
        if count == max_frequency
    ]

    return modes


def data_range(data):
    """
    Calculate range.

    Formula:
        Range = max(x) - min(x)
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    return max(data) - min(data)


# ============================================================
# 2. VARIANCE AND STANDARD DEVIATION
# ============================================================

def variance(data, sample=False):
    """
    Calculate variance.

    Population variance:
        σ² = Σ(x_i - μ)² / N

    Sample variance:
        s² = Σ(x_i - x̄)² / (n - 1)

    Parameters
    ----------
    data : list
        Numerical data.

    sample : bool
        False -> population variance
        True  -> sample variance
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    if sample and len(data) < 2:
        raise ValueError(
            "Sample variance requires at least 2 observations."
        )

    center = mean(data)

    squared_deviations = [
        (x - center) ** 2
        for x in data
    ]

    denominator = len(data) - 1 if sample else len(data)

    return sum(squared_deviations) / denominator


def standard_deviation(data, sample=False):
    """
    Calculate standard deviation.

    Formula:
        σ = sqrt(variance)
    """

    return sqrt(variance(data, sample=sample))


# ============================================================
# 3. QUARTILES AND IQR
# ============================================================

def quartiles(data):
    """
    Calculate Q1, Q2 and Q3.

    Q1 = 25th percentile
    Q2 = 50th percentile (median)
    Q3 = 75th percentile

    Uses linear interpolation.
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    sorted_data = sorted(data)
    n = len(sorted_data)

    def percentile(p):
        position = (n - 1) * p

        lower = int(position)
        upper = lower + 1

        if upper >= n:
            return sorted_data[lower]

        weight = position - lower

        return (
            sorted_data[lower]
            + weight * (sorted_data[upper] - sorted_data[lower])
        )

    q1 = percentile(0.25)
    q2 = percentile(0.50)
    q3 = percentile(0.75)

    return q1, q2, q3


def iqr(data):
    """
    Calculate Interquartile Range.

    Formula:
        IQR = Q3 - Q1
    """

    q1, _, q3 = quartiles(data)

    return q3 - q1


def detect_outliers_iqr(data):
    """
    Detect outliers using the IQR method.

    Lower bound:
        Q1 - 1.5 * IQR

    Upper bound:
        Q3 + 1.5 * IQR
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    q1, _, q3 = quartiles(data)
    interquartile_range = q3 - q1

    lower_bound = q1 - 1.5 * interquartile_range
    upper_bound = q3 + 1.5 * interquartile_range

    outliers = [
        x
        for x in data
        if x < lower_bound or x > upper_bound
    ]

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": interquartile_range,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outliers
    }


# ============================================================
# 4. COVARIANCE
# ============================================================

def covariance(x, y, sample=True):
    """
    Calculate covariance between two variables.

    Formula:

        Cov(X,Y) =
        Σ[(x_i - x̄)(y_i - ȳ)] / (n - 1)

    for sample covariance.

    Positive covariance:
        X and Y tend to increase together.

    Negative covariance:
        When X increases, Y tends to decrease.

    Near zero:
        Weak linear relationship.
    """

    if len(x) != len(y):
        raise ValueError(
            "x and y must have the same number of observations."
        )

    if len(x) < 2:
        raise ValueError(
            "At least 2 observations are required."
        )

    mean_x = mean(x)
    mean_y = mean(y)

    products = [
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    ]

    denominator = len(x) - 1 if sample else len(x)

    return sum(products) / denominator


# ============================================================
# 5. PEARSON CORRELATION
# ============================================================

def pearson_correlation(x, y):
    """
    Calculate Pearson correlation coefficient.

    Formula:

        r = Cov(X,Y) / (σ_x * σ_y)

    Range:
        -1 <= r <= 1

    Interpretation:
        r close to +1 -> strong positive relationship
        r close to -1 -> strong negative relationship
        r close to  0 -> weak linear relationship
    """

    if len(x) != len(y):
        raise ValueError(
            "x and y must have the same number of observations."
        )

    if len(x) < 2:
        raise ValueError(
            "At least 2 observations are required."
        )

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    )

    sum_squared_x = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    sum_squared_y = sum(
        (yi - mean_y) ** 2
        for yi in y
    )

    denominator = sqrt(
        sum_squared_x * sum_squared_y
    )

    if denominator == 0:
        raise ValueError(
            "Correlation is undefined when one variable has zero variance."
        )

    return numerator / denominator


# ============================================================
# 6. Z-SCORE
# ============================================================

def z_score(value, data):
    """
    Calculate z-score.

    Formula:

        z = (x - μ) / σ

    Interpretation:
        z = 0       -> value equals the mean
        z > 0       -> value is above the mean
        z < 0       -> value is below the mean
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    mu = mean(data)
    sigma = standard_deviation(data)

    if sigma == 0:
        raise ValueError(
            "Z-score is undefined when standard deviation is zero."
        )

    return (value - mu) / sigma


def z_scores(data):
    """
    Calculate z-score for every observation.
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    mu = mean(data)
    sigma = standard_deviation(data)

    if sigma == 0:
        raise ValueError(
            "Z-score is undefined when standard deviation is zero."
        )

    return [
        (x - mu) / sigma
        for x in data
    ]


# ============================================================
# 7. STANDARDIZATION
# ============================================================

def standardize(data):
    """
    Standardize data using z-score normalization.

    Formula:

        x_standardized = (x - μ) / σ

    This is commonly used in Machine Learning
    before models such as:
        - Logistic Regression
        - SVM
        - KNN
        - Neural Networks
    """

    return z_scores(data)


# ============================================================
# 8. CONFIDENCE INTERVAL
# ============================================================

def confidence_interval_mean(data, confidence=0.95):
    """
    Calculate approximate confidence interval for the population mean.

    This implementation uses the normal critical value:

        90% -> 1.645
        95% -> 1.960
        99% -> 2.576

    Formula:

        CI = mean ± z * (σ / sqrt(n))

    Returns:
        lower_bound, upper_bound
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    critical_values = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576
    }

    if confidence not in critical_values:
        raise ValueError(
            "Supported confidence levels: 0.90, 0.95, 0.99"
        )

    z = critical_values[confidence]

    mu = mean(data)
    sigma = standard_deviation(data)

    n = len(data)

    margin_error = z * (sigma / sqrt(n))

    lower_bound = mu - margin_error
    upper_bound = mu + margin_error

    return lower_bound, upper_bound


# ============================================================
# 9. HEALTHCARE PREVALENCE
# ============================================================

def prevalence(labels, positive_label=1):
    """
    Calculate prevalence of a disease/risk condition.

    Formula:

        Prevalence =
        number of positive cases / total cases

    Example:
        Risk = 1 -> positive
        Risk = 0 -> negative
    """

    if not labels:
        raise ValueError("Labels cannot be empty.")

    positive_cases = sum(
        1
        for label in labels
        if label == positive_label
    )

    total_cases = len(labels)

    return positive_cases / total_cases


def prevalence_percentage(labels, positive_label=1):
    """
    Calculate disease prevalence as percentage.
    """

    return prevalence(
        labels,
        positive_label=positive_label
    ) * 100


# ============================================================
# 10. GROUP STATISTICS
# ============================================================

def group_statistics(values, labels, target_label):
    """
    Calculate statistics for a specific group.

    Example:
        values = systolic blood pressure
        labels = Risk
        target_label = 1

    This can be used to compare:

        Risk = 0
        vs.
        Risk = 1
    """

    if len(values) != len(labels):
        raise ValueError(
            "values and labels must have the same length."
        )

    group = [
        value
        for value, label in zip(values, labels)
        if label == target_label
    ]

    if not group:
        raise ValueError(
            "No observations found for target_label."
        )

    q1, q2, q3 = quartiles(group)

    return {
        "count": len(group),
        "mean": mean(group),
        "median": q2,
        "variance": variance(group),
        "standard_deviation": standard_deviation(group),
        "minimum": min(group),
        "maximum": max(group),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1
    }


# ============================================================
# 11. DESCRIPTIVE SUMMARY
# ============================================================

def descriptive_summary(data):
    """
    Return a complete descriptive statistical summary.
    """

    if not data:
        raise ValueError("Data cannot be empty.")

    q1, q2, q3 = quartiles(data)

    return {
        "count": len(data),
        "mean": mean(data),
        "median": q2,
        "mode": mode(data),
        "variance": variance(data),
        "standard_deviation": standard_deviation(data),
        "minimum": min(data),
        "maximum": max(data),
        "range": data_range(data),
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "IQR": q3 - q1
    }


# ============================================================
# 12. TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STATISTICS - AI HEALTHCARE ASSISTANT")
    print("=" * 60)

    data = [10, 12, 12, 15, 18, 20, 21, 25, 30]

    print("\n--- DESCRIPTIVE STATISTICS ---")

    print("Mean:",
          mean(data))

    print("Median:",
          median(data))

    print("Mode:",
          mode(data))

    print("Range:",
          data_range(data))

    print("Variance:",
          variance(data))

    print("Standard deviation:",
          standard_deviation(data))

    print("\n--- QUARTILES ---")

    q1, q2, q3 = quartiles(data)

    print("Q1:", q1)
    print("Q2:", q2)
    print("Q3:", q3)
    print("IQR:", iqr(data))

    print("\n--- OUTLIER DETECTION ---")

    outlier_result = detect_outliers_iqr(data)

    print("Lower bound:",
          outlier_result["lower_bound"])

    print("Upper bound:",
          outlier_result["upper_bound"])

    print("Outliers:",
          outlier_result["outliers"])

    print("\n--- CORRELATION ---")

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 8, 10]

    print("Covariance:",
          covariance(x, y))

    print("Pearson correlation:",
          pearson_correlation(x, y))

    print("\n--- Z-SCORE ---")

    print("Z-score of 20:",
          z_score(20, data))

    print("All z-scores:",
          z_scores(data))

    print("\n--- CONFIDENCE INTERVAL ---")

    lower, upper = confidence_interval_mean(data)

    print("95% CI:")
    print("Lower:", lower)
    print("Upper:", upper)

    print("\n--- HEALTHCARE PREVALENCE ---")

    risk_labels = [0, 0, 1, 0, 1, 1, 0, 0, 1, 0]

    print("Prevalence:",
          prevalence_percentage(risk_labels),
          "%")

    print("\n--- GROUP STATISTICS ---")

    sbp = [
        120, 125, 118, 150, 155,
        160, 122, 130, 145, 158
    ]

    risk = [
        0, 0, 0, 1, 1,
        1, 0, 0, 1, 1
    ]

    risk_1_stats = group_statistics(
        sbp,
        risk,
        target_label=1
    )

    print("Risk = 1 statistics:")

    for key, value in risk_1_stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("STATISTICS MODULE TEST COMPLETED")
    print("=" * 60)