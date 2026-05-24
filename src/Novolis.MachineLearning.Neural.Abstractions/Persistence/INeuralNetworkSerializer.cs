namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>Serializes <see cref="NetworkSnapshot"/> to and from text.</summary>
public interface INeuralNetworkSerializer
{
    /// <summary>Writes a snapshot to a string.</summary>
    string Serialize(NetworkSnapshot snapshot);

    /// <summary>Parses a snapshot from a string.</summary>
    NetworkSnapshot Deserialize(string content);
}
