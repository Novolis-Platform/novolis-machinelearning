namespace Novolis.MachineLearning.Neural;

/// <summary>Neural network that supports mutation and copying.</summary>
public interface IMutableNeuralNetwork : INeuralNetwork
{
    /// <summary>Creates a deep copy, optionally with a new name.</summary>
    IMutableNeuralNetwork Clone(string? name = null);

    /// <summary>Applies random weight and bias mutations.</summary>
    void Mutate(Random random, MutationSettings settings);

    /// <summary>Copies weights and biases from another network with matching topology.</summary>
    void CopyFrom(INeuralNetwork other);
}
