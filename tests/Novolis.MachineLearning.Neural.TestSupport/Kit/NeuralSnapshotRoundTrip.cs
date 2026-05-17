using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;

namespace Novolis.MachineLearning.Neural.Kit;

/// <summary>JSON snapshot round-trip using the same serializer as production persistence.</summary>
public static class NeuralSnapshotRoundTrip
{
    private static readonly JsonNeuralNetworkSerializer Serializer = new();

    public static DenseNetwork RoundTrip(DenseNetwork network, string snapshotId = "round-trip-test")
    {
        var snapshot = network.ToSnapshot(snapshotId);
        var json = Serializer.Serialize(snapshot);
        var back = Serializer.Deserialize(json);
        return DenseNetwork.FromSnapshot(back);
    }

    public static string Serialize(DenseNetwork network, string snapshotId = "snapshot") =>
        Serializer.Serialize(network.ToSnapshot(snapshotId));
}
