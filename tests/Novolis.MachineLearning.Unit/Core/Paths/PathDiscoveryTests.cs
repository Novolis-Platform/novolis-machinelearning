using System.IO.Abstractions.TestingHelpers;

using Novolis.MachineLearning.Core.Paths;

namespace Novolis.MachineLearning.Core.Tests.Paths;

public sealed class PathDiscoveryTests
{
    [Test]
    public async Task TryGetDirectoryContainingFile_FindsMarkerInParent()
    {
        var fs = new MockFileSystem(new Dictionary<string, MockFileData>
        {
            { @"C:\repo\Novolis.MachineLearning.slnx", new MockFileData("") },
            { @"C:\repo\bin\debug\App.dll", new MockFileData("") },
        });
        var found = NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(
            @"C:\repo\bin\debug",
            "Novolis.MachineLearning.slnx",
            maxDepth: 8,
            fs);
        await Assert.That(found).IsEqualTo(@"C:\repo");
    }

    [Test]
    public async Task TryGetDirectoryContainingFile_ReturnsNull_WhenNotFound()
    {
        var fs = new MockFileSystem();
        var found = NovolisMachineLearningPathDiscovery.TryGetDirectoryContainingFile(
            @"C:\nowhere",
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
