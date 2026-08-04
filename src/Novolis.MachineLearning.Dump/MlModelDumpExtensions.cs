using System.IO.Abstractions;

using Microsoft.ML;

using Novolis.CodeGen.Reflection.Dump;
using Novolis.MachineLearning.Neural.Persistence;

namespace Novolis.MachineLearning.Dump;

/// <summary>Helpers that dump ML snapshots or export ML.NET models to dump stores.</summary>
public static class MlModelDumpExtensions
{
    /// <summary>Writes a neural <see cref="NetworkSnapshot"/> as a DumpClass <c>.cs</c> fixture.</summary>
    /// <param name="snapshot">Snapshot to dump.</param>
    /// <param name="path">Destination path.</param>
    /// <param name="fileSystem">Optional file system.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The path written.</returns>
    public static ValueTask<string> DumpToFileAsync(
        this NetworkSnapshot snapshot,
        string path,
        IFileSystem? fileSystem = null,
        CancellationToken cancellationToken = default)
        => snapshot.DumpClassToFileAsync(path, fileSystem, options: null, cancellationToken);

    /// <summary>
    /// Saves an ML.NET model zip and a companion dump of <paramref name="metadata"/> as C#.
    /// </summary>
    /// <param name="store">Zip model store.</param>
    /// <param name="mlContext">ML.NET context.</param>
    /// <param name="model">Trained transformer.</param>
    /// <param name="schema">Training schema.</param>
    /// <param name="id">Model identity.</param>
    /// <param name="metadata">Serializable metadata dumped beside the zip.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Paths to the zip and dump files.</returns>
    public static async ValueTask<(string ZipPath, string DumpPath)> SaveWithDumpAsync<TMetadata>(
        this FileMlModelStore store,
        MLContext mlContext,
        ITransformer model,
        DataViewSchema schema,
        string id,
        TMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(store);
        var zipPath = store.Save(mlContext, model, schema, id);
        var dumpPath = System.IO.Path.Combine(store.RootDirectory, $"{id}.cs");
        await metadata.DumpClassToFileAsync(dumpPath, cancellationToken: cancellationToken)
            .ConfigureAwait(false);
        return (zipPath, dumpPath);
    }
}
