namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>Persisted neural network topology and parameters.</summary>
public sealed record NetworkSnapshot(
    string Id,
    string Name,
    int InputSize,
    int OutputSize,
    int[] LayerSizes,
    LayerSnapshot[] Layers,
    DateTimeOffset CreatedAtUtc,
    IReadOnlyDictionary<string, string>? Metadata = null);
