using System.IO.Abstractions.TestingHelpers;

using Novolis.MachineLearning.Core.Paths;
using Novolis.MachineLearning.TestInfrastructure;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Core.Tests.Paths;

[NotInParallel]
public sealed class NovolisMachineLearningDataPathsTests
{
    [Test]
    public async Task NetworksDirectory_IsUnderData_AndNamedNetworks()
    {
        var root = NovolisMachineLearningDataPaths.ResolveDataRoot();
        var nets = NovolisMachineLearningDataPaths.NetworksDirectory();
        await Assert.That(Path.IsPathRooted(nets)).IsTrue();
        await Assert.That(Path.GetFileName(nets)).IsEqualTo("networks");
        await Assert.That(Path.GetFileName(Path.GetDirectoryName(nets))).IsEqualTo("data");
        await Assert.That(Path.GetFullPath(Path.Combine(root, "networks"))).IsEqualTo(nets);
    }

    [Test]
    public async Task LogsDirectory_IsUnderData_AndNamedLogs()
    {
        var root = NovolisMachineLearningDataPaths.ResolveDataRoot();
        var logs = NovolisMachineLearningDataPaths.LogsDirectory();
        await Assert.That(Path.IsPathRooted(logs)).IsTrue();
        await Assert.That(Path.GetFileName(logs)).IsEqualTo("logs");
        await Assert.That(Path.GetFileName(Path.GetDirectoryName(logs))).IsEqualTo("data");
        await Assert.That(Path.GetFullPath(Path.Combine(root, "logs"))).IsEqualTo(logs);
    }

    [Test]
    public async Task ResolveDataRoot_ReturnsRepoData_WhenEnvOverrideIsUnset()
    {
        var prev = Environment.GetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable);
        Environment.SetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable, null);
        try
        {
            var fs = MockFs.RootedAtRepoRoot();
            var resolved = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
            var expected = fs.Path.GetFullPath(fs.Path.Combine(MockFs.RepoRoot, "data"));
            await Assert.That(resolved).IsEqualTo(expected);
        }
        finally
        {
            Environment.SetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable, prev);
        }
    }

    [Test]
    public async Task ResolveDataRoot_RespectsEnvOverride()
    {
        var prev = Environment.GetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable);
        var fs = MockFs.RootedAtRepoRoot();
        var custom = MockFs.UnderRepo(fs, "custom-data");
        Environment.SetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable, custom);
        try
        {
            var resolved = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
            await Assert.That(resolved).IsEqualTo(fs.Path.GetFullPath(custom));
        }
        finally
        {
            Environment.SetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable, prev);
        }
    }
}
