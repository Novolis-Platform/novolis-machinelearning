using System.Text.Json;

using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Neural;

public class JsonNeuralNetworkSerializerTests
{
    private readonly JsonNeuralNetworkSerializer _sut = new();

    private static NetworkSnapshot CreateTestSnapshot(
        string id = "test-id",
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var net = DenseNetwork.Create("test-net", 2, [3], 1, random: new Random(42));
        return net.ToSnapshot(id, metadata);
    }

    [Test]
    public async Task RoundTrip_InputSizeIsPreserved()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.InputSize).IsEqualTo(snapshot.InputSize);
    }

    [Test]
    public async Task RoundTrip_OutputSizeIsPreserved()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.OutputSize).IsEqualTo(snapshot.OutputSize);
    }

    [Test]
    public async Task RoundTrip_LayerCountIsPreserved()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.Layers.Length).IsEqualTo(snapshot.Layers.Length);
    }

    [Test]
    public async Task RoundTrip_NetworkNameIsPreserved()
    {
        var snapshot = CreateTestSnapshot("id-name");
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.Name).IsEqualTo(snapshot.Name);
    }

    [Test]
    public async Task RoundTrip_SnapshotIdIsPreserved()
    {
        var snapshot = CreateTestSnapshot("unique-snapshot-id");
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.Id).IsEqualTo("unique-snapshot-id");
    }

    [Test]
    public async Task RoundTrip_WeightsArePreservedExactly()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);

        for (int l = 0; l < snapshot.Layers.Length; l++)
        {
            var orig = snapshot.Layers[l].Weights;
            var rest = restored.Layers[l].Weights;
            await Assert.That(rest.Length).IsEqualTo(orig.Length);
            for (int i = 0; i < orig.Length; i++)
                for (int j = 0; j < orig[i].Length; j++)
                    await Assert.That(rest[i][j]).IsEqualTo(orig[i][j]);
        }
    }

    [Test]
    public async Task RoundTrip_BiasesArePreservedExactly()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);

        for (int l = 0; l < snapshot.Layers.Length; l++)
            await Assert.That(restored.Layers[l].Biases).IsEquivalentTo(snapshot.Layers[l].Biases);
    }

    [Test]
    public async Task RoundTrip_ActivationNamesArePreserved()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);

        for (int l = 0; l < snapshot.Layers.Length; l++)
            await Assert.That(restored.Layers[l].Activation).IsEqualTo(snapshot.Layers[l].Activation);
    }

    [Test]
    public async Task RoundTrip_MetadataIsPreserved()
    {
        var meta = new Dictionary<string, string> { ["env"] = "test", ["version"] = "1.0" };
        var snapshot = CreateTestSnapshot("m-id", meta);
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);

        await Assert.That(restored.Metadata).IsNotNull();
        await Assert.That(restored.Metadata!["env"]).IsEqualTo("test");
        await Assert.That(restored.Metadata["version"]).IsEqualTo("1.0");
    }

    [Test]
    public async Task RoundTrip_NullMetadataRemainsNull()
    {
        var snapshot = CreateTestSnapshot("null-meta", null);
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.Metadata).IsNull();
    }

    [Test]
    public async Task RoundTrip_CreatedAtUtcIsPreserved()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        var restored = _sut.Deserialize(json);
        await Assert.That(restored.CreatedAtUtc).IsEqualTo(snapshot.CreatedAtUtc).Within(TimeSpan.FromMilliseconds(1));
    }

    [Test]
    public async Task Serialize_ProducesValidJson()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        await Assert.That(() => JsonDocument.Parse(json)).ThrowsNothing();
    }

    [Test]
    public async Task Serialize_ProducesHumanReadableOutput_WithNewlines()
    {
        var snapshot = CreateTestSnapshot();
        var json = _sut.Serialize(snapshot);
        await Assert.That(json).Contains("\n").Because("WriteIndented=true must produce multi-line JSON");
    }

    [Test]
    public async Task Serialize_TwoSnapshots_AreIndependent()
    {
        var s1 = CreateTestSnapshot("id-alpha");
        var s2 = CreateTestSnapshot("id-beta");
        var json1 = _sut.Serialize(s1);
        var json2 = _sut.Serialize(s2);

        await Assert.That(json1).Contains("id-alpha");
        await Assert.That(json2).Contains("id-beta");
        await Assert.That(json1).DoesNotContain("id-beta");
        await Assert.That(json2).DoesNotContain("id-alpha");
    }

    [Test]
    public async Task Deserialize_NullJsonLiteral_ThrowsInvalidOperationException()
    {
        // JsonSerializer.Deserialize<T>("null") returns null for reference types,
        // which triggers the ?? throw in the implementation.
        await Assert.That(() => _sut.Deserialize("null")).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task Deserialize_EmptyString_ThrowsJsonException()
    {
        await Assert.That(() => _sut.Deserialize("")).Throws<JsonException>();
    }
}
