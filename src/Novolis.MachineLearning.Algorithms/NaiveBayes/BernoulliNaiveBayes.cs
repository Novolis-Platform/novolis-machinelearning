namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>
/// Bernoulli Naive Bayes trainer for binary <see cref="Features{T}"/> vectors (<see cref="bool"/> only).
/// </summary>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
public sealed class BernoulliNaiveBayesTrainer<TLabel> : INaiveBayesTrainer<bool, TLabel>
    where TLabel : notnull
{
    private readonly NaiveBayesOptions _options;

    /// <summary>Creates a trainer with default <see cref="NaiveBayesOptions"/>.</summary>
    public BernoulliNaiveBayesTrainer()
        : this(new NaiveBayesOptions())
    {
    }

    /// <summary>Creates a trainer with custom options.</summary>
    /// <param name="options">Laplace smoothing and related safeguards.</param>
    public BernoulliNaiveBayesTrainer(NaiveBayesOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _options = options;
    }

    /// <inheritdoc />
    public INaiveBayesClassifier<bool, TLabel> Fit(IEnumerable<LabeledExample<bool, TLabel>> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var materialized = examples as IList<LabeledExample<bool, TLabel>> ?? examples.ToList();
        if (materialized.Count == 0)
            throw new ArgumentException("At least one labeled example is required.", nameof(examples));

        var featureCount = materialized[0].Features.Length;
        var byLabel = new Dictionary<TLabel, List<Features<bool>>>();

        foreach (var example in materialized)
        {
            if (example.Label is null)
                throw new ArgumentException("Labels must be non-null.", nameof(examples));

            if (example.Features.Length != featureCount)
            {
                throw new ArgumentException(
                    $"All feature vectors must have length {featureCount}, but found {example.Features.Length}.",
                    nameof(examples));
            }

            if (!byLabel.TryGetValue(example.Label, out var bucket))
            {
                bucket = [];
                byLabel[example.Label] = bucket;
            }

            bucket.Add(example.Features);
        }

        var total = materialized.Count;
        var classes = byLabel.Keys.OrderBy(static x => x?.ToString() ?? string.Empty, StringComparer.Ordinal).ToArray();
        var models = new BernoulliClassModel<TLabel>[classes.Length];
        var alpha = _options.Smoothing;

        for (var c = 0; c < classes.Length; c++)
        {
            var label = classes[c];
            var rows = byLabel[label];
            var trueProbabilities = new double[featureCount];

            for (var j = 0; j < featureCount; j++)
            {
                var trues = 0;
                foreach (var row in rows)
                {
                    if (row[j])
                        trues++;
                }

                // Laplace-smoothed P(x_j = true | class)
                trueProbabilities[j] = (trues + alpha) / (rows.Count + 2 * alpha);
            }

            models[c] = new BernoulliClassModel<TLabel>(
                label,
                Math.Log(rows.Count / (double)total),
                trueProbabilities);
        }

        return new BernoulliNaiveBayesClassifier<TLabel>(featureCount, models);
    }
}
