namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

/// <summary>Posterior score for one class after a Naive Bayes prediction.</summary>
/// <typeparam name="TLabel">Non-null class label type.</typeparam>
/// <param name="Label">Predicted class.</param>
/// <param name="LogScore">Unnormalized log-score used for argmax.</param>
/// <param name="Probability">Normalized posterior probability in <c>[0, 1]</c>.</param>
public readonly record struct ClassScore<TLabel>(
    TLabel Label,
    double LogScore,
    double Probability)
    where TLabel : notnull;
