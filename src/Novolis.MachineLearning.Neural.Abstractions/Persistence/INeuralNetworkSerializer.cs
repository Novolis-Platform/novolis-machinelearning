namespace Novolis.MachineLearning.Neural.Persistence;

public interface INeuralNetworkSerializer
{
    string Serialize(NetworkSnapshot snapshot);
    NetworkSnapshot Deserialize(string content);
}
