namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>Shared numeric safeguards for Naive Bayes trainers.</summary>
public sealed class NaiveBayesOptions
{
    /// <summary>
    /// Minimum feature variance for Gaussian Naive Bayes (prevents divide-by-zero).
    /// Must be positive.
    /// </summary>
    public double VarianceFloor { get; init; } = 1e-9;

    /// <summary>
    /// Additive (Laplace) smoothing for Bernoulli probability estimates.
    /// Must be positive.
    /// </summary>
    public double Smoothing { get; init; } = 1.0;

    /// <summary>Validates option ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when a value is not positive.</exception>
    public void Validate()
    {
        if (VarianceFloor <= 0)
            throw new ArgumentOutOfRangeException(nameof(VarianceFloor), VarianceFloor, "Variance floor must be positive.");

        if (Smoothing <= 0)
            throw new ArgumentOutOfRangeException(nameof(Smoothing), Smoothing, "Smoothing must be positive.");
    }
}
