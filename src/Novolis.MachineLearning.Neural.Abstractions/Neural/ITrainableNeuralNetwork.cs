namespace Novolis.MachineLearning.Neural;

/// <summary>Neural network that supports supervised and policy-gradient training steps.</summary>
public interface ITrainableNeuralNetwork : INeuralNetwork
{
    /// <summary>One supervised gradient step toward <paramref name="target"/>.</summary>
    double TrainSupervised(ReadOnlySpan<double> input, ReadOnlySpan<double> target, double learningRate);

    /// <summary>
    /// One gradient-descent step using ∂L/∂a for the last layer activations (for a linear head, same as ∂L/∂z).
    /// Used for policy-gradient style losses; same backprop as <see cref="TrainSupervised"/> with caller-supplied output deltas.
    /// </summary>
    void TrainWithOutputGradient(ReadOnlySpan<double> input, ReadOnlySpan<double> lossGradientWrtOutputActivation, double learningRate);
}
