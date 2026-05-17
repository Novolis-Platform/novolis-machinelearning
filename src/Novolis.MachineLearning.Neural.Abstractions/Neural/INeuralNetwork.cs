namespace Novolis.MachineLearning.Neural;

public interface INeuralNetwork
{
    string Name { get; }
    int InputSize { get; }
    int OutputSize { get; }
    IReadOnlyList<int> LayerSizes { get; }
    ReadOnlyMemory<double> Forward(ReadOnlySpan<double> input);
    NetworkEvaluation Evaluate(ReadOnlySpan<double> input);
}
