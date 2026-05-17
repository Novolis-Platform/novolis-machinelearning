using System.IO.Abstractions;

namespace Novolis.MachineLearning.Neural.Persistence;

public sealed class FileNeuralNetworkRepository : INeuralNetworkRepository
{
    private readonly string _rootDirectory;
    private readonly INeuralNetworkSerializer _serializer;
    private readonly IFileSystem _fileSystem;

    public FileNeuralNetworkRepository(string rootDirectory, INeuralNetworkSerializer serializer)
        : this(rootDirectory, serializer, new FileSystem())
    {
    }

    public FileNeuralNetworkRepository(string rootDirectory, INeuralNetworkSerializer serializer, IFileSystem fileSystem)
    {
        _rootDirectory = rootDirectory;
        _serializer = serializer;
        _fileSystem = fileSystem;
    }

    public async ValueTask SaveAsync(NetworkSnapshot snapshot, CancellationToken cancellationToken = default)
    {
        _fileSystem.Directory.CreateDirectory(_rootDirectory);
        var path = _fileSystem.Path.Combine(_rootDirectory, $"{snapshot.Id}.json");
        var content = _serializer.Serialize(snapshot);
        await _fileSystem.File.WriteAllTextAsync(path, content, cancellationToken);
    }

    public async ValueTask<NetworkSnapshot?> LoadAsync(string id, CancellationToken cancellationToken = default)
    {
        var path = _fileSystem.Path.Combine(_rootDirectory, $"{id}.json");
        if (!_fileSystem.File.Exists(path))
            return null;
        var content = await _fileSystem.File.ReadAllTextAsync(path, cancellationToken);
        return _serializer.Deserialize(content);
    }

    public ValueTask<IReadOnlyList<NetworkSnapshotMetadata>> ListAsync(CancellationToken cancellationToken = default)
    {
        _fileSystem.Directory.CreateDirectory(_rootDirectory);
        var items = _fileSystem.Directory
            .EnumerateFiles(_rootDirectory, "*.json", SearchOption.TopDirectoryOnly)
            .Select(_fileSystem.File.ReadAllText)
            .Select(_serializer.Deserialize)
            .Select(x => new NetworkSnapshotMetadata(x.Id, x.Name, x.CreatedAtUtc, x.Metadata))
            .OrderByDescending(x => x.CreatedAtUtc)
            .ToArray();
        return ValueTask.FromResult<IReadOnlyList<NetworkSnapshotMetadata>>(items);
    }
}
