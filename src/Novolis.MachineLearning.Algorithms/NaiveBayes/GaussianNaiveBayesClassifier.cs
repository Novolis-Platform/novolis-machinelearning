using System.Numerics;

namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>Fitted Gaussian Naive Bayes classifier.</summary>
/// <typeparam name="TFeature">Numeric unmanaged feature type.</typeparam>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
public sealed class GaussianNaiveBayesClassifier<TFeature, TLabel> : INaiveBayesClassifier<TFeature, TLabel>
    where TFeature : unmanaged, INumber<TFeature>
    where TLabel : notnull
{
    private readonly GaussianClassModel<TLabel>[] _models;

    internal GaussianNaiveBayesClassifier(int featureCount, GaussianClassModel<TLabel>[] models)
    {
        FeatureCount = featureCount;
        _models = models;
        Classes = models.Select(static m => m.Label).ToArray();
    }

    /// <inheritdoc />
    public int FeatureCount { get; }

    /// <inheritdoc />
    public IReadOnlyList<TLabel> Classes { get; }

    /// <inheritdoc />
    public TLabel Predict(Features<TFeature> features)
    {
        var scores = PredictScores(features);
        return scores.OrderByDescending(static s => s.LogScore).First().Label;
    }

    /// <inheritdoc />
    public IReadOnlyList<ClassScore<TLabel>> PredictScores(Features<TFeature> features)
    {
        EnsureFeatureCount(features);

        var logScores = new double[_models.Length];
        for (var c = 0; c < _models.Length; c++)
        {
            var model = _models[c];
            var log = model.LogPrior;
            var span = features.AsSpan();
            for (var j = 0; j < FeatureCount; j++)
                log += LogGaussianPdf(double.CreateChecked(span[j]), model.Means[j], model.Variances[j]);

            logScores[c] = log;
        }

        return NaiveBayesScoring.ToClassScores(_models.Select(static m => m.Label).ToArray(), logScores);
    }

    private void EnsureFeatureCount(Features<TFeature> features)
    {
        if (features.Length != FeatureCount)
        {
            throw new ArgumentException(
                $"Expected {FeatureCount} features, but received {features.Length}.",
                nameof(features));
        }
    }

    private static double LogGaussianPdf(double x, double mean, double variance)
    {
        var z = x - mean;
        return -0.5 * (Math.Log(2 * Math.PI * variance) + (z * z / variance));
    }
}
