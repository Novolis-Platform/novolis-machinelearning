namespace Novolis.MachineLearning.Neural.Persistence;

public interface INeuralNetworkRepository
{
    ValueTask SaveAsync(NetworkSnapshot snapshot, CancellationToken cancellationToken = default);
    ValueTask<NetworkSnapshot?> LoadAsync(string id, CancellationToken cancellationToken = default);
    ValueTask<IReadOnlyList<NetworkSnapshotMetadata>> ListAsync(CancellationToken cancellationToken = default);
}
