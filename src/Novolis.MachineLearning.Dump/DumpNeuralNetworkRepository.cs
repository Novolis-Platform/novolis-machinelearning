using System.IO.Abstractions;

using Novolis.CodeGen.Reflection.Dump;
using Novolis.MachineLearning.Neural.Persistence;

namespace Novolis.MachineLearning.Dump;

/// <summary>
/// Persists <see cref="NetworkSnapshot"/> instances as CodeGen dump <c>.cs</c> fixtures
/// plus a JSON sidecar for runtime round-trip load.
/// </summary>
public sealed class DumpNeuralNetworkRepository : INeuralNetworkRepository
{
    private readonly DumpFileStore _dumpStore;
    private readonly INeuralNetworkSerializer _serializer;
    private readonly IFileSystem _fileSystem;
    private readonly string _rootDirectory;

    /// <summary>Creates a repository using the physical file system and JSON serializer.</summary>
    /// <param name="rootDirectory">Directory for <c>.cs</c> dumps and <c>.json</c> sidecars.</param>
    public DumpNeuralNetworkRepository(string rootDirectory)
        : this(rootDirectory, new JsonNeuralNetworkSerializer(), new FileSystem())
    {
    }

    /// <summary>Creates a repository with injectable serializer and file system.</summary>
    /// <param name="rootDirectory">Directory for dump artifacts.</param>
    /// <param name="serializer">JSON snapshot serializer for load/list.</param>
    /// <param name="fileSystem">File system abstraction.</param>
    public DumpNeuralNetworkRepository(
        string rootDirectory,
        INeuralNetworkSerializer serializer,
        IFileSystem fileSystem)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);
        ArgumentNullException.ThrowIfNull(serializer);
        ArgumentNullException.ThrowIfNull(fileSystem);
        _rootDirectory = rootDirectory;
        _serializer = serializer;
        _fileSystem = fileSystem;
        _dumpStore = new DumpFileStore(rootDirectory, fileSystem);
    }

    /// <summary>Root directory for dump and JSON files.</summary>
    public string RootDirectory => _rootDirectory;

    /// <inheritdoc />
    public async ValueTask SaveAsync(NetworkSnapshot snapshot, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        _fileSystem.Directory.CreateDirectory(_rootDirectory);

        await _dumpStore.SaveClassAsync(snapshot.Id, snapshot, cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        var jsonPath = JsonPath(snapshot.Id);
        var content = _serializer.Serialize(snapshot);
        await _fileSystem.File.WriteAllTextAsync(jsonPath, content, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    public async ValueTask<NetworkSnapshot?> LoadAsync(string id, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        var path = JsonPath(id);
        if (!_fileSystem.File.Exists(path))
            return null;
        var content = await _fileSystem.File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
        return _serializer.Deserialize(content);
    }

    /// <summary>Loads the dumped C# fixture source for <paramref name="id"/>, or null when missing.</summary>
    /// <param name="id">Snapshot id.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>C# dump source text.</returns>
    public ValueTask<string?> LoadDumpSourceAsync(string id, CancellationToken cancellationToken = default)
        => _dumpStore.LoadSourceAsync(id, cancellationToken);

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

    private string JsonPath(string id)
        => _fileSystem.Path.Combine(_rootDirectory, $"{id}.json");
}
