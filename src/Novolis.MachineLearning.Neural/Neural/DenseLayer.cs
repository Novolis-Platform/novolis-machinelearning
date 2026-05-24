namespace Novolis.MachineLearning.Neural;

/// <summary>Weights, biases, and activation for one fully-connected layer.</summary>
public sealed class DenseLayer
{
    /// <summary>Input dimension.</summary>
    public required int InputCount { get; init; }

    /// <summary>Output dimension.</summary>
    public required int OutputCount { get; init; }

    /// <summary>Activation applied after the affine transform.</summary>
    public required ActivationKind Activation { get; init; }

    /// <summary>Weight matrix [input, output].</summary>
    public required double[,] Weights { get; init; }

    /// <summary>Bias vector per output neuron.</summary>
    public required double[] Biases { get; init; }
}
