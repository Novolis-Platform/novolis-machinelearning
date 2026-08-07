namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>Fitted Naive Bayes model that scores and predicts labels from <see cref="Features{T}"/>.</summary>
/// <typeparam name="TFeature">Unmanaged feature element type.</typeparam>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
public interface INaiveBayesClassifier<TFeature, TLabel>
    where TFeature : unmanaged, IEquatable<TFeature>
    where TLabel : notnull
{
    /// <summary>Feature dimensionality expected by <see cref="Predict"/>.</summary>
    int FeatureCount { get; }

    /// <summary>Distinct class labels learned during training, in stable order.</summary>
    IReadOnlyList<TLabel> Classes { get; }

    /// <summary>Returns the highest-scoring class for <paramref name="features"/>.</summary>
    /// <param name="features">Feature vector of length <see cref="FeatureCount"/>.</param>
    /// <returns>Predicted label.</returns>
    /// <exception cref="ArgumentException">Thrown when the feature length does not match.</exception>
    TLabel Predict(Features<TFeature> features);

    /// <summary>Returns per-class log-scores and normalized probabilities.</summary>
    /// <param name="features">Feature vector of length <see cref="FeatureCount"/>.</param>
    /// <returns>One score entry per learned class.</returns>
    /// <exception cref="ArgumentException">Thrown when the feature length does not match.</exception>
    IReadOnlyList<ClassScore<TLabel>> PredictScores(Features<TFeature> features);
}
