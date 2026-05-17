namespace Novolis.MachineLearning.Neural.Persistence;

public sealed record NetworkSnapshotMetadata(
    string Id,
    string Name,
    DateTimeOffset CreatedAtUtc,
    IReadOnlyDictionary<string, string>? Metadata);
