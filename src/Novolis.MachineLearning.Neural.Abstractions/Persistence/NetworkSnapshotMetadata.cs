namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>Lightweight listing entry for a saved network.</summary>
public sealed record NetworkSnapshotMetadata(
    string Id,
    string Name,
    DateTimeOffset CreatedAtUtc,
    IReadOnlyDictionary<string, string>? Metadata);
