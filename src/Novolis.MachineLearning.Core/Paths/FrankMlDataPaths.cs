using System.IO.Abstractions;

using Novolis.MachineLearning.Core.IO;

namespace Novolis.MachineLearning.Core.Paths;

/// <summary>Canonical on-disk <c>data</c> layout shared by Web, Avalonia, CLI, and Aspire.</summary>
public static class NovolisMachineLearningDataPaths
{
    /// <summary>Absolute path to the shared <c>data</c> directory (contains <c>networks</c>, <c>logs</c>, etc.).</summary>
    public const string DataRootEnvironmentVariable = "NOVOLIS_ML_DATA";

    /// <summary>
    /// Resolves the <c>data</c> directory: <see cref="DataRootEnvironmentVariable"/> if set; else <c>data</c> next to
    /// <c>Novolis.MachineLearning.slnx</c> at the compile-time repo root (<see cref="NovolisFileSystem.RepoRoot"/>).
    /// </summary>
    public static string ResolveDataRoot(IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        var fromEnv = Environment.GetEnvironmentVariable(DataRootEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(fromEnv))
            return fs.Path.GetFullPath(fromEnv.Trim());

        return fs.Path.GetFullPath(fs.Path.Combine(NovolisFileSystem.RepoRoot, "data"));
    }

    /// <summary>Network snapshot JSON directory (<c>data/networks</c>).</summary>
    public static string NetworksDirectory(IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        return fs.Path.GetFullPath(fs.Path.Combine(ResolveDataRoot(fs), "networks"));
    }

    /// <summary>Application log directory (<c>data/logs</c>).</summary>
    public static string LogsDirectory(IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        return fs.Path.GetFullPath(fs.Path.Combine(ResolveDataRoot(fs), "logs"));
    }
}
