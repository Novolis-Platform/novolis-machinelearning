namespace Novolis.MachineLearning.Neural.Kit;

/// <summary>Thrown by framework-agnostic kit checks when a numeric or structural expectation fails.</summary>
public sealed class NeuralAssertionException : Exception
{
    public NeuralAssertionException(string message) : base(message)
    {
    }
}
