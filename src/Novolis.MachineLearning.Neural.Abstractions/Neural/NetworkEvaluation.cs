namespace Novolis.MachineLearning.Neural;

public sealed record NetworkEvaluation(
    double[] Output,
    double[][] Activations,
    double[][] WeightedSums);
