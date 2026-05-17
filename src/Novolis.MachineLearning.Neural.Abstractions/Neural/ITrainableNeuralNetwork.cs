namespace Novolis.MachineLearning.Neural;

public interface ITrainableNeuralNetwork : INeuralNetwork
{
    double TrainSupervised(ReadOnlySpan<double> input, ReadOnlySpan<double> target, double learningRate);

    /// <summary>
    /// One gradient-descent step using ∂L/∂a for the last layer activations (for a linear head, same as ∂L/∂z).
    /// Used for policy-gradient style losses; same backprop as <see cref="TrainSupervised"/> with caller-supplied output deltas.
    /// </summary>
    void TrainWithOutputGradient(ReadOnlySpan<double> input, ReadOnlySpan<double> lossGradientWrtOutputActivation, double learningRate);
}
