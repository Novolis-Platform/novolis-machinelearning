using System.Text.Json;

namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>JSON serializer for <see cref="NetworkSnapshot"/>.</summary>
public sealed class JsonNeuralNetworkSerializer : INeuralNetworkSerializer
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    /// <inheritdoc />
    public string Serialize(NetworkSnapshot snapshot)
        => JsonSerializer.Serialize(snapshot, Options);

    /// <inheritdoc />
    public NetworkSnapshot Deserialize(string content)
        => JsonSerializer.Deserialize<NetworkSnapshot>(content)
           ?? throw new InvalidOperationException("Could not deserialize network snapshot.");
}
