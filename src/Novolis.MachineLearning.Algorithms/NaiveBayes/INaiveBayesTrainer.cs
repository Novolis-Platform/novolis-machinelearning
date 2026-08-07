namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>Fits a Naive Bayes classifier from labeled <see cref="Features{T}"/> examples.</summary>
/// <typeparam name="TFeature">Unmanaged feature element type.</typeparam>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
public interface INaiveBayesTrainer<TFeature, TLabel>
    where TFeature : unmanaged, IEquatable<TFeature>
    where TLabel : notnull
{
    /// <summary>Estimates class priors and per-feature likelihood parameters.</summary>
    /// <param name="examples">Non-empty training set with a consistent feature length.</param>
    /// <returns>A fitted classifier.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="examples"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when examples are empty or lengths disagree.</exception>
    INaiveBayesClassifier<TFeature, TLabel> Fit(IEnumerable<LabeledExample<TFeature, TLabel>> examples);
}
