namespace Novolis.MachineLearning.Algorithms;

/// <summary>
/// Fixed-length feature vector contract used by classic algorithms such as Naive Bayes.
/// </summary>
/// <typeparam name="T">
/// Feature value type. Must be an unmanaged equality-comparable value
/// (for example <see cref="bool"/>, <see cref="int"/>, or <see cref="double"/>).
/// </typeparam>
public interface IFeatures<T>
    where T : unmanaged, IEquatable<T>
{
    /// <summary>Number of feature dimensions.</summary>
    int Length { get; }

    /// <summary>Feature value at <paramref name="index"/>.</summary>
    /// <param name="index">Zero-based feature index.</param>
    /// <returns>The feature value.</returns>
    T this[int index] { get; }

    /// <summary>Exposes the underlying values without copying.</summary>
    /// <returns>A span over the feature values.</returns>
    ReadOnlySpan<T> AsSpan();
}
