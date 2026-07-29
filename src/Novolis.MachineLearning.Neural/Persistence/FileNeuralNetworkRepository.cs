using System.IO.Abstractions;

namespace Novolis.MachineLearning.Neural.Persistence;

/// <summary>File-system implementation of <see cref="INeuralNetworkRepository"/>.</summary>
public sealed class FileNeuralNetworkRepository : INeuralNetworkRepository
{
    private readonly string _rootDirectory;
    private readonly INeuralNetworkSerializer _serializer;
    private readonly IFileSystem _fileSystem;

    /// <summary>Creates a repository using the physical file system.</summary>
    public FileNeuralNetworkRepository(string rootDirectory, INeuralNetworkSerializer serializer)
        : this(rootDirectory, serializer, new FileSystem())
    {
    }

    /// <summary>Creates a repository with an injectable file system (for tests).</summary>
    public FileNeuralNetworkRepository(string rootDirectory, INeuralNetworkSerializer serializer, IFileSystem fileSystem)
    {
        _rootDirectory = rootDirectory;
        _serializer = serializer;
        _fileSystem = fileSystem;
    }

    /// <inheritdoc />
    public async ValueTask SaveAsync(NetworkSnapshot snapshot, CancellationToken cancellationToken = default)
    {
        _fileSystem.Directory.CreateDirectory(_rootDirectory);
        var path = _fileSystem.Path.Combine(_rootDirectory, $"{snapshot.Id}.json");
        var content = _serializer.Serialize(snapshot);
        await _fileSystem.File.WriteAllTextAsync(path, content, cancellationToken);
    }

    /// <inheritdoc />
    public async ValueTask<NetworkSnapshot?> LoadAsync(string id, CancellationToken cancellationToken = default)
    {
        var path = _fileSystem.Path.Combine(_rootDirectory, $"{id}.json");
        if (!_fileSystem.File.Exists(path))
            return null;
        var content = await _fileSystem.File.ReadAllTextAsync(path, cancellationToken);
        return _serializer.Deserialize(content);
    }

    /// <inheritdoc />
    public ValueTask<IReadOnlyList<NetworkSnapshotMetadata>> ListAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _fileSystem.Directory.CreateDirectory(_rootDirectory);

        var list = new List<NetworkSnapshotMetadata>();
        foreach (var file in _fileSystem.Directory.EnumerateFiles(_rootDirectory, "*.json"))
        {
            var content = _fileSystem.File.ReadAllText(file);
            var snapshot = _serializer.Deserialize(content);
            list.Add(new NetworkSnapshotMetadata(snapshot.Id, snapshot.Name, snapshot.CreatedAtUtc, snapshot.Metadata));
        }

        list.Sort(static (a, b) => b.CreatedAtUtc.CompareTo(a.CreatedAtUtc));
        return ValueTask.FromResult<IReadOnlyList<NetworkSnapshotMetadata>>(list);
    }
}
