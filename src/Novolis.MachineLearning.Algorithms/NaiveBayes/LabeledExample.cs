namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>One supervised training row: features plus a class label.</summary>
/// <typeparam name="TFeature">Unmanaged feature element type.</typeparam>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
/// <param name="Features">Feature vector.</param>
/// <param name="Label">Ground-truth class.</param>
public readonly record struct LabeledExample<TFeature, TLabel>(
    Features<TFeature> Features,
    TLabel Label)
    where TFeature : unmanaged, IEquatable<TFeature>
    where TLabel : notnull;
