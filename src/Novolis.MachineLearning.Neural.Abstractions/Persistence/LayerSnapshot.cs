namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>Serialized weights for one dense layer.</summary>
public sealed record LayerSnapshot(
    int InputCount,
    int OutputCount,
    string Activation,
    double[][] Weights,
    double[] Biases);
