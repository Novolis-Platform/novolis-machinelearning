namespace Novolis.MachineLearning.Neural;

/// <summary>Gaussian mutation rates for evolutionary training.</summary>
/// <param name="WeightMutationRate">Probability of mutating each weight.</param>
/// <param name="WeightMutationSigma">Standard deviation of weight noise.</param>
/// <param name="BiasMutationRate">Probability of mutating each bias.</param>
/// <param name="BiasMutationSigma">Standard deviation of bias noise.</param>
public sealed record MutationSettings(
    double WeightMutationRate,
    double WeightMutationSigma,
    double BiasMutationRate,
    double BiasMutationSigma);
