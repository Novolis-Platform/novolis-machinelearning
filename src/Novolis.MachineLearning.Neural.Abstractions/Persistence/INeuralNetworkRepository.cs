namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>Persists <see cref="NetworkSnapshot"/> instances by id.</summary>
public interface INeuralNetworkRepository
{
    /// <summary>Saves or overwrites a snapshot.</summary>
    ValueTask SaveAsync(NetworkSnapshot snapshot, CancellationToken cancellationToken = default);

    /// <summary>Loads a snapshot by id, or null when missing.</summary>
    ValueTask<NetworkSnapshot?> LoadAsync(string id, CancellationToken cancellationToken = default);

    /// <summary>Lists snapshot metadata without loading full weights.</summary>
    ValueTask<IReadOnlyList<NetworkSnapshotMetadata>> ListAsync(CancellationToken cancellationToken = default);
}
