namespace Novolis.MachineLearning.Neural.Persistence;

public sealed record LayerSnapshot(
    int InputCount,
    int OutputCount,
    string Activation,
    double[][] Weights,
    double[] Biases);
