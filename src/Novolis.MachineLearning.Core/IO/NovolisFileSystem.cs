using System.IO.Abstractions;

namespace Novolis.MachineLearning.Core.IO;

/// <summary>
/// Repo-rooted entry points for <see cref="IFileSystem"/>.
/// </summary>
public static class NovolisFileSystem
{
    /// <summary>Solution file name used to locate the repository root.</summary>
    public const string SolutionFileName = "Novolis.MachineLearning.slnx";

    /// <summary>Repository root (directory that contains <see cref="SolutionFileName"/>).</summary>
    public static string RepoRoot => FindRepoRoot();

    /// <summary>A new physical <see cref="FileSystem"/> instance backed by the real OS file system.</summary>
    public static IFileSystem CreatePhysical() => new FileSystem();

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, SolutionFileName)))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
