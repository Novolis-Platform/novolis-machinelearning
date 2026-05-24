namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>Named preset bundling a <see cref="NetworkSnapshot"/> for UI or seeding.</summary>
public sealed record NeuralPreset(
    string Id,
    string Name,
    string Family,
    string ColorName,
    NetworkSnapshot Snapshot,
    IReadOnlyDictionary<string, string>? Metadata = null);
