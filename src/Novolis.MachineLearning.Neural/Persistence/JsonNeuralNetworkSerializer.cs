using System.Text.Json;

namespace Novolis.MachineLearning.Neural.Persistence;

public sealed class JsonNeuralNetworkSerializer : INeuralNetworkSerializer
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    public string Serialize(NetworkSnapshot snapshot)
        => JsonSerializer.Serialize(snapshot, Options);

    public NetworkSnapshot Deserialize(string content)
        => JsonSerializer.Deserialize<NetworkSnapshot>(content)
           ?? throw new InvalidOperationException("Could not deserialize network snapshot.");
}
