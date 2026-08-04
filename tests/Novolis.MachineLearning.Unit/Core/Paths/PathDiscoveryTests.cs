using System.IO.Abstractions.TestingHelpers;

using Novolis.MachineLearning.Core.Paths;

namespace Novolis.MachineLearning.Core.Tests.Paths;

public sealed class PathDiscoveryTests
{
    static string Root(params string[] parts)
    {
        var root = Path.GetFullPath(Path.Combine(Path.DirectorySeparatorChar == '\\' ? @"C:\repo" : "/repo"));
        return parts.Length == 0 ? root : Path.Combine([root, .. parts]);
    }

    [Test]
    public async Task TryGetDirectoryContainingFile_FindsMarkerInParent()
    {
        var repo = Root();
        var marker = Root("Novolis.MachineLearning.slnx");
        var start = Root("bin", "debug");
        var fs = new MockFileSystem(new Dictionary<string, MockFileData>
        {
            { marker, new MockFileData("") },
            { Root("bin", "debug", "App.dll"), new MockFileData("") },
        }, repo);
        var found = NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(
            start,
            "Novolis.MachineLearning.slnx",
            maxDepth: 8,
            fs);
        await Assert.That(found).IsEqualTo(repo);
    }

    [Test]
    public async Task TryGetDirectoryContainingFile_ReturnsNull_WhenNotFound()
    {
        var nowhere = Root("nowhere");
        var fs = new MockFileSystem(new Dictionary<string, MockFileData>(), nowhere);
        var found = NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(
            nowhere,
            "missing.marker",
            maxDepth: 3,
            fs);
        await Assert.That(found).IsNull();
    }

    [Test]
    public async Task TryGetDirectoryContainingFile_SwallowsInvalidPaths()
    {
        var fs = new MockFileSystem();
        var found = NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(
            ":::not-a-path",
            "x",
            maxDepth: 1,
            fs);
        await Assert.That(found).IsNull();
    }
}
