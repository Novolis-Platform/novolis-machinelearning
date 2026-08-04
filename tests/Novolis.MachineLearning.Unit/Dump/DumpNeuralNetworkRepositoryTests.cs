using System.IO.Abstractions.TestingHelpers;

using Novolis.MachineLearning.Dump;
using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;
using Novolis.MachineLearning.TestInfrastructure;

namespace Novolis.MachineLearning.Dump.Tests;

public sealed class DumpNeuralNetworkRepositoryTests
{
    private readonly MockFileSystem _fs = MockFs.RootedAtRepoRoot();
    private readonly string _rootDir;
    private readonly DumpNeuralNetworkRepository _sut;

    public DumpNeuralNetworkRepositoryTests()
    {
        _rootDir = MockFs.UnderRepo(_fs, "tmp", "ml-dump-repo-" + Guid.NewGuid().ToString("N"));
        _sut = new DumpNeuralNetworkRepository(_rootDir, new JsonNeuralNetworkSerializer(), _fs);
    }

    [Test]
    public async Task SaveAsync_WritesDumpCs_AndJsonSidecar()
    {
        var net = DenseNetwork.Create("policy", 2, [3], 1, random: new Random(7));
        var snapshot = net.ToSnapshot("best-policy");

        await _sut.SaveAsync(snapshot);

        await Assert.That(_fs.File.Exists(_fs.Path.Combine(_rootDir, "best-policy.json"))).IsTrue();
        await Assert.That(_fs.File.Exists(_fs.Path.Combine(_rootDir, "best-policy.cs"))).IsTrue();

        var loaded = await _sut.LoadAsync("best-policy");
        await Assert.That(loaded).IsNotNull();
        await Assert.That(loaded!.Id).IsEqualTo("best-policy");

        var dump = await _sut.LoadDumpSourceAsync("best-policy");
        await Assert.That(dump).IsNotNullOrWhiteSpace();
        await Assert.That(dump!).Contains("NetworkSnapshot");
    }

    [Test]
    public async Task ListAsync_ReturnsMetadataFromJson()
    {
        var net = DenseNetwork.Create("n", 2, [2], 1, random: new Random(1));
        await _sut.SaveAsync(net.ToSnapshot("a"));
        await _sut.SaveAsync(net.ToSnapshot("b"));

        var list = await _sut.ListAsync();
        await Assert.That(list.Count).IsEqualTo(2);
    }
}
