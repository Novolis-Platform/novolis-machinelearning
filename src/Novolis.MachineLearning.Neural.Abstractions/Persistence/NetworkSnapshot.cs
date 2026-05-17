namespace Novolis.MachineLearning.Neural.Persistence;

public sealed record NetworkSnapshot(
    string Id,
    string Name,
    int InputSize,
    int OutputSize,
    int[] LayerSizes,
    LayerSnapshot[] Layers,
    DateTimeOffset CreatedAtUtc,
    IReadOnlyDictionary<string, string>? Metadata = null);
