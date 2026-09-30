namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>Fitted Bernoulli Naive Bayes classifier over <see cref="Features{T}"/> of <see cref="bool"/>.</summary>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
public sealed class BernoulliNaiveBayesClassifier<TLabel> : INaiveBayesClassifier<bool, TLabel>
    where TLabel : notnull
{
    private readonly BernoulliClassModel<TLabel>[] _models;

    internal BernoulliNaiveBayesClassifier(int featureCount, BernoulliClassModel<TLabel>[] models)
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
    public TLabel Predict(Features<bool> features)
    {
        var scores = PredictScores(features);
        return scores.OrderByDescending(static s => s.LogScore).First().Label;
    }

    /// <inheritdoc />
    public IReadOnlyList<ClassScore<TLabel>> PredictScores(Features<bool> features)
    {
        if (features.Length != FeatureCount)
        {
            throw new ArgumentException(
                $"Expected {FeatureCount} features, but received {features.Length}.",
                nameof(features));
        }

        var logScores = new double[_models.Length];
        for (var c = 0; c < _models.Length; c++)
        {
            var model = _models[c];
            var log = model.LogPrior;
            var span = features.AsSpan();
            for (var j = 0; j < FeatureCount; j++)
            {
                var pTrue = model.TrueProbabilities[j];
                var p = span[j] ? pTrue : 1.0 - pTrue;
                log += Math.Log(p);
            }

            logScores[c] = log;
        }

        return NaiveBayesScoring.ToClassScores(_models.Select(static m => m.Label).ToArray(), logScores);
    }
}
