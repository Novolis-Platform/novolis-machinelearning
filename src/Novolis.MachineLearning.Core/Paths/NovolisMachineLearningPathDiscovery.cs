using System.IO.Abstractions;

using Novolis.MachineLearning.Core.IO;

namespace Novolis.MachineLearning.Core.Paths;

/// <summary>Walks parent directories from a start path to find a marker file (repo / layout discovery).</summary>
public static class NovolisMachineLearningPathDiscovery
{
    /// <summary>Returns the directory that contains <paramref name="fileName"/>, or null if not found within <paramref name="maxDepth"/>.</summary>
    public static string? TryGetDirectoryContainingFile(
        string startDir,
        string fileName,
        int maxDepth = 24,
        IFileSystem? fileSystem = null)
    {
        var fs = fileSystem ?? NovolisFileSystem.CreatePhysical();
        try
        {
            var dir = fs.DirectoryInfo.New(fs.Path.GetFullPath(startDir));
            for (var depth = 0; depth < maxDepth && dir is not null; depth++)
            {
                if (fs.File.Exists(fs.Path.Combine(dir.FullName, fileName)))
                    return dir.FullName;
                dir = dir.Parent;
            }
        }
        catch
        {
            // Best-effort only.
        }

        return null;
    }
}
