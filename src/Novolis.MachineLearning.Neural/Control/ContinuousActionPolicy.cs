namespace Novolis.MachineLearning.Neural;

/// <summary>
/// Continuous-control policy wrapping a feed-forward network.
/// Observation → action vector in approximately [-1, 1] (tanh-friendly).
/// </summary>
public sealed class ContinuousActionPolicy
{
    private readonly double[] _scratchOut;

    /// <summary>Wraps an existing network as a continuous policy.</summary>
    public ContinuousActionPolicy(INeuralNetwork network)
    {
        ArgumentNullException.ThrowIfNull(network);
        if (network.OutputSize < 1)
            throw new ArgumentException("Network must have at least one output.", nameof(network));
        Network = network;
        _scratchOut = new double[network.OutputSize];
    }

    /// <summary>Underlying feed-forward network.</summary>
    public INeuralNetwork Network { get; }

    /// <summary>Expected observation vector length.</summary>
    public int ObservationSize => Network.InputSize;

    /// <summary>Action vector length.</summary>
    public int ActionSize => Network.OutputSize;

    /// <summary>Creates a tanh-hidden dense policy suitable for arcade stick imitation.</summary>
    public static ContinuousActionPolicy Create(
        string name,
        int observationSize,
        int actionSize,
        int[]? hiddenSizes = null,
        Random? random = null)
    {
        var net = DenseNetwork.Create(
            name,
            observationSize,
            hiddenSizes ?? [24, 16],
            actionSize,
            ActivationKind.Tanh,
            random);
        return new ContinuousActionPolicy(net);
    }

    /// <summary>Forward pass; writes clamped actions into <paramref name="actions"/>.</summary>
    public void Act(ReadOnlySpan<double> observation, Span<double> actions, double clamp = 1.0)
    {
        if (observation.Length != ObservationSize)
            throw new ArgumentException($"Expected observation size {ObservationSize}.", nameof(observation));
        if (actions.Length < ActionSize)
            throw new ArgumentException($"Actions buffer must be at least {ActionSize}.", nameof(actions));

        var output = Network.Forward(observation).Span;
        for (var i = 0; i < ActionSize; i++)
            actions[i] = Math.Clamp(output[i], -clamp, clamp);
    }

    /// <summary>Supervised imitation step when the underlying network is trainable.</summary>
    public double Imitate(
        ReadOnlySpan<double> observation,
        ReadOnlySpan<double> targetActions,
        double learningRate)
    {
        if (Network is not ITrainableNeuralNetwork trainable)
            throw new InvalidOperationException("Underlying network is not trainable.");
        if (observation.Length != ObservationSize)
            throw new ArgumentException($"Expected observation size {ObservationSize}.", nameof(observation));
        if (targetActions.Length < ActionSize)
            throw new ArgumentException($"Target actions must be at least {ActionSize}.", nameof(targetActions));

        for (var i = 0; i < ActionSize; i++)
            _scratchOut[i] = targetActions[i];
        return trainable.TrainSupervised(observation, _scratchOut.AsSpan(0, ActionSize), learningRate);
    }
}
