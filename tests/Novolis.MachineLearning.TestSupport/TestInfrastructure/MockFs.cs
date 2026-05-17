using System.IO.Abstractions;
using System.IO.Abstractions.TestingHelpers;

namespace Novolis.MachineLearning.TestInfrastructure;

/// <summary>
/// Factory helpers for <see cref="MockFileSystem"/> instances rooted at the Novolis.MachineLearning repo.
/// </summary>
public static class MockFs
{
    public static string RepoRoot => FindRepoRoot();

    public static MockFileSystem RootedAtRepoRoot() => RootedAtRepoRoot(seed: null);

    public static MockFileSystem RootedAtRepoRoot(IDictionary<string, MockFileData>? seed)
    {
        var files = seed ?? new Dictionary<string, MockFileData>();
        return new MockFileSystem(files, currentDirectory: RepoRoot);
    }

    public static string UnderRepo(IFileSystem fileSystem, params string[] relativeParts)
    {
        var combined = new string[relativeParts.Length + 1];
        combined[0] = RepoRoot;
        Array.Copy(relativeParts, 0, combined, 1, relativeParts.Length);
        return fileSystem.Path.Combine(combined);
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Novolis.MachineLearning.slnx")))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
