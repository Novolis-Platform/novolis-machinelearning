namespace Novolis.MachineLearning.Neural;

public sealed record MutationSettings(
    double WeightMutationRate,
    double WeightMutationSigma,
    double BiasMutationRate,
    double BiasMutationSigma);
