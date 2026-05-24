using System.IO.Abstractions;
using System.IO.Abstractions.TestingHelpers;

using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;
using Novolis.MachineLearning.TestInfrastructure;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Neural;

public class FileNeuralNetworkRepositoryTests
{
    private readonly MockFileSystem _fs = MockFs.RootedAtRepoRoot();
    private readonly string _rootDir;
    private readonly JsonNeuralNetworkSerializer _serializer = new();
    private readonly FileNeuralNetworkRepository _sut;

    public FileNeuralNetworkRepositoryTests()
    {
        _rootDir = MockFs.UnderRepo(_fs, "tmp", "frank-ml-repo-tests-" + Guid.NewGuid().ToString("N"));
        _sut = new FileNeuralNetworkRepository(_rootDir, _serializer, _fs);
    }

    private static NetworkSnapshot CreateSnapshot(
        string id,
        string name = "test-net",
        DateTimeOffset? createdAt = null,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var net = DenseNetwork.Create(name, 2, [3], 1, random: new Random(42));
        var snapshot = net.ToSnapshot(id, metadata);
        return createdAt.HasValue ? snapshot with { CreatedAtUtc = createdAt.Value } : snapshot;
    }

    [Test]
    public async Task SaveAsync_CreatesRootDirectoryIfNotExists()
    {
        await Assert.That(_fs.Directory.Exists(_rootDir)).IsFalse().Because("dir must not exist before first save");
        await _sut.SaveAsync(CreateSnapshot("id-1"));
        await Assert.That(_fs.Directory.Exists(_rootDir)).IsTrue();
    }

    [Test]
    public async Task SaveAsync_WritesFileNamedWithSnapshotId()
    {
        await _sut.SaveAsync(CreateSnapshot("my-snapshot-id"));
        var expectedPath = _fs.Path.Combine(_rootDir, "my-snapshot-id.json");
        await Assert.That(_fs.File.Exists(expectedPath)).IsTrue();
    }

    [Test]
    public async Task SaveAsync_TwiceWithSameId_OverwritesPreviousFile()
    {
        var first = CreateSnapshot("dup-id") with { CreatedAtUtc = DateTimeOffset.UtcNow.AddHours(-1) };
        var second = CreateSnapshot("dup-id") with { CreatedAtUtc = DateTimeOffset.UtcNow };
        await _sut.SaveAsync(first);
        await _sut.SaveAsync(second);

        var loaded = await _sut.LoadAsync("dup-id");
        await Assert.That(loaded).IsNotNull();
        await Assert.That(loaded!.CreatedAtUtc).IsEqualTo(second.CreatedAtUtc).Within(TimeSpan.FromMilliseconds(1));
    }

    [Test]
    public async Task SaveAsync_CancelledToken_ThrowsOperationCanceledException()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.That(async () => await _sut.SaveAsync(CreateSnapshot("cancel-save"), cts.Token))
            .Throws<OperationCanceledException>();
    }

    [Test]
    public async Task LoadAsync_ReturnsCorrectSnapshot()
    {
        var snapshot = CreateSnapshot("load-me");
        await _sut.SaveAsync(snapshot);
        var loaded = await _sut.LoadAsync("load-me");
        await Assert.That(loaded).IsNotNull();
        await Assert.That(loaded!.Id).IsEqualTo("load-me");
        await Assert.That(loaded.Name).IsEqualTo(snapshot.Name);
    }

    [Test]
    public async Task LoadAsync_NonExistentId_ReturnsNull()
    {
        var result = await _sut.LoadAsync("does-not-exist");
        await Assert.That(result).IsNull();
    }

    [Test]
    public async Task LoadAsync_EmptyDirectory_ReturnsNull()
    {
        _fs.Directory.CreateDirectory(_rootDir);
        var result = await _sut.LoadAsync("any-id");
        await Assert.That(result).IsNull();
    }

    [Test]
    public async Task LoadAsync_MultipleSaved_EachByIdReturnsCorrectSnapshot()
    {
        var s1 = CreateSnapshot("s1", "net-a");
        var s2 = CreateSnapshot("s2", "net-b");
        var s3 = CreateSnapshot("s3", "net-c");
        await _sut.SaveAsync(s1);
        await _sut.SaveAsync(s2);
        await _sut.SaveAsync(s3);

        var l1 = await _sut.LoadAsync("s1");
        var l2 = await _sut.LoadAsync("s2");
        var l3 = await _sut.LoadAsync("s3");

        await Assert.That(l1!.Name).IsEqualTo("net-a");
        await Assert.That(l2!.Name).IsEqualTo("net-b");
        await Assert.That(l3!.Name).IsEqualTo("net-c");
    }

    [Test]
    public async Task LoadAsync_CancelledToken_ThrowsOperationCanceledException()
    {
        var snapshot = CreateSnapshot("cancel-load");
        await _sut.SaveAsync(snapshot);

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.That(async () => await _sut.LoadAsync("cancel-load", cts.Token))
            .Throws<OperationCanceledException>();
    }

    [Test]
    public async Task ListAsync_EmptyDirectory_ReturnsEmptyList()
    {
        _fs.Directory.CreateDirectory(_rootDir);
        var list = await _sut.ListAsync();
        await Assert.That(list).IsEmpty();
    }

    [Test]
    public async Task ListAsync_CreatesDirectoryIfNotExists()
    {
        await Assert.That(_fs.Directory.Exists(_rootDir)).IsFalse();
        var list = await _sut.ListAsync();
        await Assert.That(_fs.Directory.Exists(_rootDir)).IsTrue();
        await Assert.That(list).IsEmpty();
    }

    [Test]
    public async Task ListAsync_MultipleSaved_ReturnsAllSnapshots()
    {
        await _sut.SaveAsync(CreateSnapshot("la-1", "n1"));
        await _sut.SaveAsync(CreateSnapshot("la-2", "n2"));
        await _sut.SaveAsync(CreateSnapshot("la-3", "n3"));

        var list = await _sut.ListAsync();
        await Assert.That(list).Count().IsEqualTo(3);
        await Assert.That(list.Select(x => x.Id).ToArray()).IsEquivalentTo(new[] { "la-1", "la-2", "la-3" });
    }

    [Test]
    public async Task ListAsync_ReturnsOrderedByCreatedAtUtcDescending()
    {
        var baseTime = DateTimeOffset.UtcNow;
        await _sut.SaveAsync(CreateSnapshot("oldest", createdAt: baseTime));
        await _sut.SaveAsync(CreateSnapshot("middle", createdAt: baseTime.AddSeconds(1)));
        await _sut.SaveAsync(CreateSnapshot("newest", createdAt: baseTime.AddSeconds(2)));

        var list = await _sut.ListAsync();
        await Assert.That(list).Count().IsEqualTo(3);
        await Assert.That(list[0].Id).IsEqualTo("newest");
        await Assert.That(list[1].Id).IsEqualTo("middle");
        await Assert.That(list[2].Id).IsEqualTo("oldest");
    }

    [Test]
    public async Task RoundTrip_SaveLoad_ForwardOutputMatchesOriginal()
    {
        var net = DenseNetwork.Create("rt-net", 2, [3], 1, random: new Random(42));
        double[] input = [1.0, 2.0];
        var expectedOutput = net.Forward(input).ToArray();

        var snapshot = net.ToSnapshot("rt-roundtrip");
        await _sut.SaveAsync(snapshot);
        var loaded = await _sut.LoadAsync("rt-roundtrip");
        await Assert.That(loaded).IsNotNull();

        var restoredNet = DenseNetwork.FromSnapshot(loaded!);
        var restoredOutput = restoredNet.Forward(input).ToArray();

        await Assert.That(restoredOutput.Length).IsEqualTo(expectedOutput.Length);
        for (int i = 0; i < expectedOutput.Length; i++)
            await Assert.That(restoredOutput[i]).IsEqualTo(expectedOutput[i]).Within(1e-9);
    }

    [Test]
    public async Task RoundTrip_SnapshotNamePreservedThroughSaveLoad()
    {
        var snapshot = CreateSnapshot("name-test", "very-special-name");
        await _sut.SaveAsync(snapshot);
        var loaded = await _sut.LoadAsync("name-test");
        await Assert.That(loaded!.Name).IsEqualTo("very-special-name");
    }

    [Test]
    public async Task RoundTrip_MetadataPreservedThroughSaveLoad()
    {
        var meta = new Dictionary<string, string> { ["epoch"] = "500", ["loss"] = "0.01" };
        var snapshot = CreateSnapshot("meta-test", metadata: meta);
        await _sut.SaveAsync(snapshot);
        var loaded = await _sut.LoadAsync("meta-test");
        await Assert.That(loaded!.Metadata).IsNotNull();
        await Assert.That(loaded.Metadata!["epoch"]).IsEqualTo("500");
        await Assert.That(loaded.Metadata["loss"]).IsEqualTo("0.01");
    }

    [Test]
    public async Task SaveLoad_LargeNetwork_DataIsPreserved()
    {
        var net = DenseNetwork.Create("large", 50, [128, 64], 20, random: new Random(42));
        var snapshot = net.ToSnapshot("large-net");
        await _sut.SaveAsync(snapshot);

        var loaded = await _sut.LoadAsync("large-net");
        await Assert.That(loaded).IsNotNull();
        await Assert.That(loaded!.InputSize).IsEqualTo(50);
        await Assert.That(loaded.OutputSize).IsEqualTo(20);
        await Assert.That(loaded.Layers.Length).IsEqualTo(3);
        await Assert.That(loaded.Layers[0].Weights.Length).IsEqualTo(50);
        await Assert.That(loaded.Layers[0].Weights[0].Length).IsEqualTo(128);
    }

    [Test]
    public async Task SaveAsync_ConcurrentSavesDifferentIds_AllSucceed()
    {
        var tasks = Enumerable.Range(0, 8)
            .Select(i => CreateSnapshot($"concurrent-{i}", $"net-{i}"))
            .Select(s => _sut.SaveAsync(s).AsTask())
            .ToArray();
        await Task.WhenAll(tasks);

        var list = await _sut.ListAsync();
        await Assert.That(list).Count().IsEqualTo(8);
        await Assert.That(list.Select(x => x.Id).ToArray()).IsEquivalentTo(
            Enumerable.Range(0, 8).Select(i => $"concurrent-{i}").ToArray());
    }
}
