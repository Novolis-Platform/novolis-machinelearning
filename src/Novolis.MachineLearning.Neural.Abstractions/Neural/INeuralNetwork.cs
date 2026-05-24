namespace Novolis.MachineLearning.Neural;

/// <summary>Feed-forward neural network contract.</summary>
public interface INeuralNetwork
{
    /// <summary>Display name.</summary>
    string Name { get; }

    /// <summary>Input vector size.</summary>
    int InputSize { get; }

    /// <summary>Output vector size.</summary>
    int OutputSize { get; }

    /// <summary>Layer widths including input and output.</summary>
    IReadOnlyList<int> LayerSizes { get; }

    /// <summary>Runs a forward pass and returns output activations.</summary>
    ReadOnlyMemory<double> Forward(ReadOnlySpan<double> input);

    /// <summary>Runs forward pass and returns full layer diagnostics.</summary>
    NetworkEvaluation Evaluate(ReadOnlySpan<double> input);
}
