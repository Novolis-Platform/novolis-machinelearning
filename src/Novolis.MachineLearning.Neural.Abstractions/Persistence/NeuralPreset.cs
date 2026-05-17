namespace Novolis.MachineLearning.Neural.Persistence;

public sealed record NeuralPreset(
    string Id,
    string Name,
    string Family,
    string ColorName,
    NetworkSnapshot Snapshot,
    IReadOnlyDictionary<string, string>? Metadata = null);
