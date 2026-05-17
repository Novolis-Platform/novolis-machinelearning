namespace Novolis.MachineLearning.Neural;

public interface IMutableNeuralNetwork : INeuralNetwork
{
    IMutableNeuralNetwork Clone(string? name = null);
    void Mutate(Random random, MutationSettings settings);
    void CopyFrom(INeuralNetwork other);
}
