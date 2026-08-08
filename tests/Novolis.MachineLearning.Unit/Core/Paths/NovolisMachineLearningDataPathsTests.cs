using Novolis.MachineLearning.Core.Paths;
using Novolis.MachineLearning.TestInfrastructure;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Core.Tests.Paths;

/// <summary>
/// Env override tests mutate process environment; named constraint keeps them
/// from racing each other without blocking unrelated suites ([NotInParallel] alone does).
/// </summary>
[NotInParallel("novolis-ml-data-env")]
public sealed class NovolisMachineLearningDataPathsTests
{
    [Test]
    public async Task NetworksDirectory_IsUnderData_AndNamedNetworks()
    {
        using var _ = EnvOverrideScope.Clear();
        var fs = MockFs.RootedAtRepoRoot();
        var root = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
        var nets = NovolisMachineLearningDataPaths.NetworksDirectory(fs);
        await Assert.That(Path.IsPathRooted(nets)).IsTrue();
        await Assert.That(Path.GetFileName(nets)).IsEqualTo("networks");
        await Assert.That(Path.GetFileName(Path.GetDirectoryName(nets))).IsEqualTo("data");
        await Assert.That(Path.GetFullPath(Path.Combine(root, "networks"))).IsEqualTo(nets);
    }

    [Test]
    public async Task LogsDirectory_IsUnderData_AndNamedLogs()
    {
        using var _ = EnvOverrideScope.Clear();
        var fs = MockFs.RootedAtRepoRoot();
        var root = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
        var logs = NovolisMachineLearningDataPaths.LogsDirectory(fs);
        await Assert.That(Path.IsPathRooted(logs)).IsTrue();
        await Assert.That(Path.GetFileName(logs)).IsEqualTo("logs");
        await Assert.That(Path.GetFileName(Path.GetDirectoryName(logs))).IsEqualTo("data");
        await Assert.That(Path.GetFullPath(Path.Combine(root, "logs"))).IsEqualTo(logs);
    }

    [Test]
    public async Task ResolveDataRoot_ReturnsRepoData_WhenEnvOverrideIsUnset()
    {
        using var _ = EnvOverrideScope.Clear();
        var fs = MockFs.RootedAtRepoRoot();
        var resolved = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
        var expected = fs.Path.GetFullPath(fs.Path.Combine(MockFs.RepoRoot, "data"));
        await Assert.That(resolved).IsEqualTo(expected);
    }

    [Test]
    public async Task ResolveDataRoot_RespectsEnvOverride()
    {
        var fs = MockFs.RootedAtRepoRoot();
        var custom = MockFs.UnderRepo(fs, "custom-data");
        using var _ = EnvOverrideScope.Set(custom);
        var resolved = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
        await Assert.That(resolved).IsEqualTo(fs.Path.GetFullPath(custom));
    }

    private sealed class EnvOverrideScope : IDisposable
    {
        private readonly string? _previous;

        private EnvOverrideScope(string? value)
        {
            _previous = Environment.GetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable);
            Environment.SetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable, value);
        }

        public static EnvOverrideScope Clear() => new(null);

        public static EnvOverrideScope Set(string value) => new(value);

        public void Dispose() =>
            Environment.SetEnvironmentVariable(NovolisMachineLearningDataPaths.DataRootEnvironmentVariable, _previous);
    }
}
