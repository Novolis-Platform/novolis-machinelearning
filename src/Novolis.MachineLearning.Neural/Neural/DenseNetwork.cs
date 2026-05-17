using System;

using Novolis.MachineLearning.Neural.Persistence;

namespace Novolis.MachineLearning.Neural;

public sealed class DenseNetwork : ITrainableNeuralNetwork, IMutableNeuralNetwork
{
    public required string Name { get; init; }
    public required DenseLayer[] Layers { get; init; }

    public int InputSize => Layers[0].InputCount;
    public int OutputSize => Layers[^1].OutputCount;
    public IReadOnlyList<int> LayerSizes => [InputSize, .. Layers.Select(x => x.OutputCount)];

    public static DenseNetwork Create(
        string name,
        int inputSize,
        int[] hiddenSizes,
        int outputSize,
        ActivationKind activation = ActivationKind.Tanh,
        Random? random = null)
    {
        random ??= Random.Shared;
        var allSizes = new int[hiddenSizes.Length + 2];
        allSizes[0] = inputSize;
        for (int i = 0; i < hiddenSizes.Length; i++)
            allSizes[i + 1] = hiddenSizes[i];
        allSizes[^1] = outputSize;

        var layers = new DenseLayer[allSizes.Length - 1];
        for (int l = 0; l < layers.Length; l++)
        {
            int inCount = allSizes[l];
            int outCount = allSizes[l + 1];
            double stddev = Math.Sqrt(2.0 / (inCount + outCount));
            var weights = new double[inCount, outCount];
            for (int i = 0; i < inCount; i++)
                for (int j = 0; j < outCount; j++)
                    weights[i, j] = NextGaussian(random) * stddev;
            var biases = new double[outCount];
            var act = (l == layers.Length - 1) ? ActivationKind.Linear : activation;
            layers[l] = new DenseLayer
            {
                InputCount = inCount,
                OutputCount = outCount,
                Activation = act,
                Weights = weights,
                Biases = biases
            };
        }

        return new DenseNetwork { Name = name, Layers = layers };
    }

    public ReadOnlyMemory<double> Forward(ReadOnlySpan<double> input)
        => Evaluate(input).Output;

    public NetworkEvaluation Evaluate(ReadOnlySpan<double> input)
    {
        int numLayers = Layers.Length;
        var activations = new double[numLayers + 1][];
        var weightedSums = new double[numLayers][];

        activations[0] = input.ToArray();

        for (int l = 0; l < numLayers; l++)
        {
            var layer = Layers[l];
            var z = new double[layer.OutputCount];
            var prevActivations = activations[l];

            for (int j = 0; j < layer.OutputCount; j++)
            {
                double sum = layer.Biases[j];
                for (int i = 0; i < layer.InputCount; i++)
                    sum += prevActivations[i] * layer.Weights[i, j];
                z[j] = sum;
            }

            weightedSums[l] = z;
            var a = new double[layer.OutputCount];
            for (int j = 0; j < layer.OutputCount; j++)
                a[j] = ApplyActivation(layer.Activation, z[j]);
            activations[l + 1] = a;
        }

        return new NetworkEvaluation(activations[numLayers], activations, weightedSums);
    }

    public double TrainSupervised(ReadOnlySpan<double> input, ReadOnlySpan<double> target, double learningRate)
    {
        var eval = Evaluate(input);
        int numLayers = Layers.Length;

        var deltas = new double[numLayers][];
        for (int l = 0; l < numLayers; l++)
            deltas[l] = new double[Layers[l].OutputCount];

        // Output layer delta
        int last = numLayers - 1;
        var outputActivations = eval.Activations[numLayers];
        for (int j = 0; j < Layers[last].OutputCount; j++)
            deltas[last][j] = (outputActivations[j] - target[j]) * ActivationDerivative(Layers[last].Activation, eval.WeightedSums[last][j]);

        // Backpropagate
        for (int l = last - 1; l >= 0; l--)
        {
            var nextLayer = Layers[l + 1];
            for (int j = 0; j < Layers[l].OutputCount; j++)
            {
                double error = 0.0;
                for (int k = 0; k < nextLayer.OutputCount; k++)
                    error += deltas[l + 1][k] * nextLayer.Weights[j, k];
                deltas[l][j] = error * ActivationDerivative(Layers[l].Activation, eval.WeightedSums[l][j]);
            }
        }

        // Update weights and biases
        for (int l = 0; l < numLayers; l++)
        {
            var layer = Layers[l];
            var inputToLayer = eval.Activations[l];
            for (int j = 0; j < layer.OutputCount; j++)
            {
                layer.Biases[j] -= learningRate * deltas[l][j];
                for (int i = 0; i < layer.InputCount; i++)
                    layer.Weights[i, j] -= learningRate * deltas[l][j] * inputToLayer[i];
            }
        }

        // Return MSE
        double mse = 0.0;
        for (int j = 0; j < outputActivations.Length; j++)
        {
            double diff = outputActivations[j] - target[j];
            mse += diff * diff;
        }
        return mse / outputActivations.Length;
    }

    public void TrainWithOutputGradient(ReadOnlySpan<double> input, ReadOnlySpan<double> lossGradientWrtOutputActivation, double learningRate)
    {
        if (lossGradientWrtOutputActivation.Length != OutputSize)
            throw new ArgumentException($"Expected length {OutputSize}, got {lossGradientWrtOutputActivation.Length}.", nameof(lossGradientWrtOutputActivation));

        var eval = Evaluate(input);
        var numLayers = Layers.Length;
        var deltas = new double[numLayers][];
        for (var l = 0; l < numLayers; l++)
            deltas[l] = new double[Layers[l].OutputCount];

        var last = numLayers - 1;
        for (var j = 0; j < Layers[last].OutputCount; j++)
        {
            deltas[last][j] = lossGradientWrtOutputActivation[j] *
                              ActivationDerivative(Layers[last].Activation, eval.WeightedSums[last][j]);
        }

        for (var l = last - 1; l >= 0; l--)
        {
            var nextLayer = Layers[l + 1];
            for (var j = 0; j < Layers[l].OutputCount; j++)
            {
                double error = 0.0;
                for (var k = 0; k < nextLayer.OutputCount; k++)
                    error += deltas[l + 1][k] * nextLayer.Weights[j, k];
                deltas[l][j] = error * ActivationDerivative(Layers[l].Activation, eval.WeightedSums[l][j]);
            }
        }

        for (var l = 0; l < numLayers; l++)
        {
            var layer = Layers[l];
            var inputToLayer = eval.Activations[l];
            for (var j = 0; j < layer.OutputCount; j++)
            {
                layer.Biases[j] -= learningRate * deltas[l][j];
                for (var i = 0; i < layer.InputCount; i++)
                    layer.Weights[i, j] -= learningRate * deltas[l][j] * inputToLayer[i];
            }
        }
    }

    public IMutableNeuralNetwork Clone(string? name = null)
    {
        var newLayers = new DenseLayer[Layers.Length];
        for (int l = 0; l < Layers.Length; l++)
        {
            var src = Layers[l];
            newLayers[l] = new DenseLayer
            {
                InputCount = src.InputCount,
                OutputCount = src.OutputCount,
                Activation = src.Activation,
                Weights = (double[,])src.Weights.Clone(),
                Biases = (double[])src.Biases.Clone()
            };
        }
        return new DenseNetwork { Name = name ?? Name, Layers = newLayers };
    }

    public void Mutate(Random random, MutationSettings settings)
    {
        foreach (var layer in Layers)
        {
            for (int i = 0; i < layer.InputCount; i++)
                for (int j = 0; j < layer.OutputCount; j++)
                    if (random.NextDouble() < settings.WeightMutationRate)
                        layer.Weights[i, j] += NextGaussian(random) * settings.WeightMutationSigma;

            for (int j = 0; j < layer.OutputCount; j++)
                if (random.NextDouble() < settings.BiasMutationRate)
                    layer.Biases[j] += NextGaussian(random) * settings.BiasMutationSigma;
        }
    }

    public void CopyFrom(INeuralNetwork other)
    {
        if (other.InputSize != InputSize || other.OutputSize != OutputSize)
            throw new InvalidOperationException("Network shapes do not match.");

        if (other.LayerSizes.Count != LayerSizes.Count)
            throw new InvalidOperationException("Network layer counts do not match.");

        for (int i = 0; i < LayerSizes.Count; i++)
            if (LayerSizes[i] != other.LayerSizes[i])
                throw new InvalidOperationException($"Layer size mismatch at index {i}.");

        if (other is not DenseNetwork otherDense)
            throw new InvalidOperationException("Can only copy from another DenseNetwork.");

        for (int l = 0; l < Layers.Length; l++)
        {
            if (Layers[l].Activation != otherDense.Layers[l].Activation)
                throw new InvalidOperationException($"Activation kind mismatch at layer {l}.");

            var src = otherDense.Layers[l];
            var dst = Layers[l];
            for (int i = 0; i < dst.InputCount; i++)
                for (int j = 0; j < dst.OutputCount; j++)
                    dst.Weights[i, j] = src.Weights[i, j];
            for (int j = 0; j < dst.OutputCount; j++)
                dst.Biases[j] = src.Biases[j];
        }
    }

    public NetworkSnapshot ToSnapshot(string id, IReadOnlyDictionary<string, string>? metadata = null)
    {
        var layerSnapshots = Layers.Select(l =>
        {
            var jagged = new double[l.InputCount][];
            for (int i = 0; i < l.InputCount; i++)
            {
                jagged[i] = new double[l.OutputCount];
                for (int j = 0; j < l.OutputCount; j++)
                    jagged[i][j] = l.Weights[i, j];
            }
            return new LayerSnapshot(l.InputCount, l.OutputCount, l.Activation.ToString(), jagged, l.Biases);
        }).ToArray();

        return new NetworkSnapshot(
            id,
            Name,
            InputSize,
            OutputSize,
            Layers.Select(l => l.OutputCount).ToArray(),
            layerSnapshots,
            DateTimeOffset.UtcNow,
            metadata);
    }

    public static DenseNetwork FromSnapshot(NetworkSnapshot snapshot)
    {
        var layers = snapshot.Layers.Select(l =>
        {
            if (l.Weights.Length != l.InputCount)
                throw new InvalidOperationException($"Expected {l.InputCount} weight rows but found {l.Weights.Length}.");
            if (l.Weights.Any(row => row.Length != l.OutputCount))
                throw new InvalidOperationException($"Weight row length does not match OutputCount {l.OutputCount}.");
            if (l.Biases.Length != l.OutputCount)
                throw new InvalidOperationException($"Expected {l.OutputCount} biases but found {l.Biases.Length}.");
            if (!Enum.TryParse<ActivationKind>(l.Activation, out var activation))
                throw new InvalidOperationException($"Unknown activation kind: {l.Activation}.");

            var weights = new double[l.InputCount, l.OutputCount];
            for (int i = 0; i < l.InputCount; i++)
                for (int j = 0; j < l.OutputCount; j++)
                    weights[i, j] = l.Weights[i][j];

            return new DenseLayer
            {
                InputCount = l.InputCount,
                OutputCount = l.OutputCount,
                Activation = activation,
                Weights = weights,
                Biases = l.Biases
            };
        }).ToArray();

        return new DenseNetwork { Name = snapshot.Name, Layers = layers };
    }

    private static double ApplyActivation(ActivationKind kind, double x) => kind switch
    {
        ActivationKind.Tanh => Math.Tanh(x),
        ActivationKind.Relu => Math.Max(0.0, x),
        ActivationKind.Sigmoid => 1.0 / (1.0 + Math.Exp(-x)),
        ActivationKind.Linear => x,
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null)
    };

    private static double ActivationDerivative(ActivationKind kind, double x) => kind switch
    {
        ActivationKind.Tanh => 1.0 - Math.Pow(Math.Tanh(x), 2),
        ActivationKind.Relu => x > 0 ? 1.0 : 0.0,
        ActivationKind.Sigmoid => Sigmoid(x) * (1.0 - Sigmoid(x)),
        ActivationKind.Linear => 1.0,
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null)
    };

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
