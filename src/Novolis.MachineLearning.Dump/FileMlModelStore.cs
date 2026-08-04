using System.IO.Abstractions;

using Microsoft.ML;

namespace Novolis.MachineLearning.Dump;

/// <summary>Persists ML.NET <see cref="ITransformer"/> pipelines as zip model files.</summary>
public sealed class FileMlModelStore
{
    private readonly string _rootDirectory;
    private readonly IFileSystem _fileSystem;

    /// <summary>Creates a store using the physical file system.</summary>
    /// <param name="rootDirectory">Directory for <c>.zip</c> model files.</param>
    public FileMlModelStore(string rootDirectory)
        : this(rootDirectory, new FileSystem())
    {
    }

    /// <summary>Creates a store with an injectable file system.</summary>
    /// <param name="rootDirectory">Directory for <c>.zip</c> model files.</param>
    /// <param name="fileSystem">File system abstraction.</param>
    public FileMlModelStore(string rootDirectory, IFileSystem fileSystem)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);
        ArgumentNullException.ThrowIfNull(fileSystem);
        _rootDirectory = rootDirectory;
        _fileSystem = fileSystem;
    }

    /// <summary>Absolute root directory for model zip files.</summary>
    public string RootDirectory => _rootDirectory;

    /// <summary>Resolves the zip path for an id.</summary>
    /// <param name="id">Model identity (file name without extension).</param>
    /// <returns>Full path to <c>{id}.zip</c>.</returns>
    public string GetPath(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return _fileSystem.Path.Combine(_rootDirectory, $"{id}.zip");
    }

    /// <summary>Saves <paramref name="model"/> and its schema to a zip file.</summary>
    /// <param name="mlContext">ML.NET context.</param>
    /// <param name="model">Trained transformer.</param>
    /// <param name="schema">Input schema used when training.</param>
    /// <param name="id">Model identity.</param>
    /// <returns>The path written.</returns>
    public string Save(MLContext mlContext, ITransformer model, DataViewSchema schema, string id)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentException.ThrowIfNullOrWhiteSpace(id);

        _fileSystem.Directory.CreateDirectory(_rootDirectory);
        var path = GetPath(id);
        using var stream = _fileSystem.File.Create(path);
        mlContext.Model.Save(model, schema, stream);
        return path;
    }

    /// <summary>Loads a previously saved model zip.</summary>
    /// <param name="mlContext">ML.NET context.</param>
    /// <param name="id">Model identity.</param>
    /// <returns>Loaded transformer.</returns>
    /// <exception cref="FileNotFoundException">When the zip is missing.</exception>
    public ITransformer Load(MLContext mlContext, string id)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentException.ThrowIfNullOrWhiteSpace(id);

        var path = GetPath(id);
        if (!_fileSystem.File.Exists(path))
            throw new FileNotFoundException($"ML model '{id}' was not found.", path);

        using var stream = _fileSystem.File.OpenRead(path);
        return mlContext.Model.Load(stream, out _);
    }

    /// <summary>Tries to load a model zip.</summary>
    /// <param name="mlContext">ML.NET context.</param>
    /// <param name="id">Model identity.</param>
    /// <param name="model">Loaded transformer when successful.</param>
    /// <returns><see langword="true"/> when the file existed and loaded.</returns>
    public bool TryLoad(MLContext mlContext, string id, out ITransformer? model)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentException.ThrowIfNullOrWhiteSpace(id);

        var path = GetPath(id);
        if (!_fileSystem.File.Exists(path))
        {
            model = null;
            return false;
        }

        using var stream = _fileSystem.File.OpenRead(path);
        model = mlContext.Model.Load(stream, out _);
        return true;
    }

    /// <summary>Lists model ids (file names without <c>.zip</c>).</summary>
    /// <returns>Model identities present on disk.</returns>
    public IReadOnlyList<string> ListIds()
    {
        _fileSystem.Directory.CreateDirectory(_rootDirectory);
        var ids = new List<string>();
        foreach (var file in _fileSystem.Directory.EnumerateFiles(_rootDirectory, "*.zip"))
        {
            var name = _fileSystem.Path.GetFileNameWithoutExtension(file);
            if (!string.IsNullOrWhiteSpace(name))
                ids.Add(name);
        }

        ids.Sort(StringComparer.OrdinalIgnoreCase);
        return ids;
    }

    /// <summary>Deletes a model zip when present.</summary>
    /// <param name="id">Model identity.</param>
    /// <returns><see langword="true"/> when a file was deleted.</returns>
    public bool Delete(string id)
    {
        var path = GetPath(id);
        if (!_fileSystem.File.Exists(path))
            return false;
        _fileSystem.File.Delete(path);
        return true;
    }
}
