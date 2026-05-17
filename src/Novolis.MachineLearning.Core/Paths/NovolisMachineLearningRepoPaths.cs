using System.IO.Abstractions;

using Novolis.MachineLearning.Core.IO;

namespace Novolis.MachineLearning.Core.Paths;

/// <summary>Repo-root-relative paths for local dev artifacts (Neural Labs logs, settings). Falls back under <see cref="NovolisMachineLearningDataPaths.ResolveDataRoot"/> when no repo marker is found.</summary>
public static class NovolisMachineLearningRepoPaths
{
    /// <summary>Resolves the repository root by walking up from <paramref name="startDir"/> for <c>Novolis.MachineLearning.slnx</c> or <c>Novolis.MachineLearning.sln</c>.</summary>
    public static string? TryGetRepoRoot(string? startDir = null, IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        var dir = string.IsNullOrWhiteSpace(startDir) ? AppContext.BaseDirectory : startDir;
        return NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(dir, "Novolis.MachineLearning.slnx", 24, fs)
            ?? NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(dir, "Novolis.MachineLearning.sln", 24, fs);
    }

    /// <summary><c>logs/neural-labs/runner</c> under repo root, or <c>data/logs/neural-labs/runner</c> when no repo.</summary>
    public static string NeuralLabsRunnerSessionsRoot(IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        var repo = TryGetRepoRoot(null, fs);
        if (repo is not null)
            return fs.Path.GetFullPath(fs.Path.Combine(repo, "logs", "neural-labs", "runner"));
        return fs.Path.GetFullPath(fs.Path.Combine(NovolisMachineLearningDataPaths.ResolveDataRoot(fs), "logs", "neural-labs", "runner"));
    }

    /// <summary>Exports when no active session directory (e.g. before first log session).</summary>
    public static string NeuralLabsRunnerOrphanExportsRoot(IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        var repo = TryGetRepoRoot(null, fs);
        if (repo is not null)
            return fs.Path.GetFullPath(fs.Path.Combine(repo, "logs", "neural-labs", "orphan-exports"));
        return fs.Path.GetFullPath(fs.Path.Combine(NovolisMachineLearningDataPaths.ResolveDataRoot(fs), "logs", "neural-labs", "orphan-exports"));
    }

    /// <summary><c>settings/neural-labs</c> under repo root, or <c>data/settings/neural-labs</c> when no repo.</summary>
    public static string NeuralLabsSettingsDirectory(IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        var repo = TryGetRepoRoot(null, fs);
        if (repo is not null)
            return fs.Path.GetFullPath(fs.Path.Combine(repo, "settings", "neural-labs"));
        return fs.Path.GetFullPath(fs.Path.Combine(NovolisMachineLearningDataPaths.ResolveDataRoot(fs), "settings", "neural-labs"));
    }

    /// <summary>Best-effort path relative to repo root for display/manifest; returns <paramref name="absolutePath"/> if not under repo.</summary>
    public static string TryGetPathRelativeToRepo(string absolutePath, IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        var repo = TryGetRepoRoot(null, fs);
        if (repo is null)
            return absolutePath;
        try
        {
            var rel = fs.Path.GetRelativePath(repo, absolutePath);
            if (rel.StartsWith(".." + fs.Path.DirectorySeparatorChar, StringComparison.Ordinal)
                || rel.Equals("..", StringComparison.Ordinal))
                return absolutePath;
            return rel;
        }
        catch
        {
            return absolutePath;
        }
    }
}
