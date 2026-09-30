namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

internal sealed class BernoulliClassModel<TLabel>(
    TLabel label,
    double logPrior,
    double[] trueProbabilities)
    where TLabel : notnull
{
    public TLabel Label { get; } = label;
    public double LogPrior { get; } = logPrior;
    public double[] TrueProbabilities { get; } = trueProbabilities;
}
