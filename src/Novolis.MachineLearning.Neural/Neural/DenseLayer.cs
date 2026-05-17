namespace Novolis.MachineLearning.Neural;

public sealed class DenseLayer
{
    public required int InputCount { get; init; }
    public required int OutputCount { get; init; }
    public required ActivationKind Activation { get; init; }
    public required double[,] Weights { get; init; }
    public required double[] Biases { get; init; }
}
