using System.Numerics;

namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>
/// Gaussian Naive Bayes trainer for continuous <see cref="Features{T}"/> vectors.
/// Feature elements must implement <see cref="INumber{T}"/> so they can be converted to <see cref="double"/>.
/// </summary>
/// <typeparam name="TFeature">Numeric unmanaged feature type.</typeparam>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
public sealed class GaussianNaiveBayesTrainer<TFeature, TLabel> : INaiveBayesTrainer<TFeature, TLabel>
    where TFeature : unmanaged, INumber<TFeature>
    where TLabel : notnull
{
    private readonly NaiveBayesOptions _options;

    /// <summary>Creates a trainer with default <see cref="NaiveBayesOptions"/>.</summary>
    public GaussianNaiveBayesTrainer()
        : this(new NaiveBayesOptions())
    {
    }

    /// <summary>Creates a trainer with custom options.</summary>
    /// <param name="options">Variance floor and related safeguards.</param>
    public GaussianNaiveBayesTrainer(NaiveBayesOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _options = options;
    }

    /// <inheritdoc />
    public INaiveBayesClassifier<TFeature, TLabel> Fit(IEnumerable<LabeledExample<TFeature, TLabel>> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var materialized = examples as IList<LabeledExample<TFeature, TLabel>> ?? examples.ToList();
        if (materialized.Count == 0)
            throw new ArgumentException("At least one labeled example is required.", nameof(examples));

        var featureCount = materialized[0].Features.Length;
        if (featureCount == 0)
            throw new ArgumentException("Feature vectors must be non-empty.", nameof(examples));

        var byLabel = new Dictionary<TLabel, List<Features<TFeature>>>();
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
        var models = new GaussianClassModel<TLabel>[classes.Length];

        for (var c = 0; c < classes.Length; c++)
        {
            var label = classes[c];
            var rows = byLabel[label];
            var means = new double[featureCount];
            var variances = new double[featureCount];

            for (var j = 0; j < featureCount; j++)
            {
                double sum = 0;
                foreach (var row in rows)
                    sum += ToDouble(row[j]);

                var mean = sum / rows.Count;
                double sumSq = 0;
                foreach (var row in rows)
                {
                    var delta = ToDouble(row[j]) - mean;
                    sumSq += delta * delta;
                }

                // Population variance with floor so a single-sample class still predicts.
                var variance = rows.Count == 1
                    ? _options.VarianceFloor
                    : Math.Max(sumSq / rows.Count, _options.VarianceFloor);

                means[j] = mean;
                variances[j] = variance;
            }

            models[c] = new GaussianClassModel<TLabel>(
                label,
                Math.Log(rows.Count / (double)total),
                means,
                variances);
        }

        return new GaussianNaiveBayesClassifier<TFeature, TLabel>(featureCount, models);
    }

    private static double ToDouble(TFeature value) => double.CreateChecked(value);
}
