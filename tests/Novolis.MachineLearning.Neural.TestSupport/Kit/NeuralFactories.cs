using Novolis.MachineLearning.Neural;

namespace Novolis.MachineLearning.Neural.Kit;

/// <summary>Deterministic construction helpers that delegate to public <see cref="DenseNetwork"/> / <see cref="DenseLayer"/> APIs.</summary>
public static class NeuralFactories
{
    public static DenseLayer MakeLayer(
        int inCount,
        int outCount,
        double[,] weights,
        double[] biases,
        ActivationKind activation) =>
        new()
        {
            InputCount = inCount,
            OutputCount = outCount,
            Activation = activation,
            Weights = weights,
            Biases = biases
        };

    public static DenseNetwork MakeNetwork(string name, params DenseLayer[] layers) =>
        new() { Name = name, Layers = layers };

    /// <inheritdoc cref="DenseNetwork.Create(string, int, int[], int, ActivationKind, Random?)"/>
    public static DenseNetwork CreateDense(
        string name,
        int inputSize,
        int[] hiddenSizes,
        int outputSize,
        ActivationKind activation = ActivationKind.Tanh,
        int? seed = null) =>
        DenseNetwork.Create(
            name,
            inputSize,
            hiddenSizes,
            outputSize,
            activation,
            seed.HasValue ? new Random(seed.Value) : null);
}
