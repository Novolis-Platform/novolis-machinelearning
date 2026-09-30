using System.Numerics;

namespace Novolis.MachineLearning.Algorithms.NaiveBayes;

internal sealed class GaussianClassModel<TLabel>(
    TLabel label,
    double logPrior,
    double[] means,
    double[] variances)
    where TLabel : notnull
{
    public TLabel Label { get; } = label;
    public double LogPrior { get; } = logPrior;
    public double[] Means { get; } = means;
    public double[] Variances { get; } = variances;
}
