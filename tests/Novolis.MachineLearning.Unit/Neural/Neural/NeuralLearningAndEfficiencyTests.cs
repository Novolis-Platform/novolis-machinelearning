using System.Diagnostics;

using Novolis.MachineLearning.Neural;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Neural;

/// <summary>Learnability (error reduction / function fit) and training throughput bounds.</summary>
public sealed class NeuralLearningAndEfficiencyTests : NeuralNetworkBaseTest
{
    /// <summary>A single linear head can fit a known affine map from few samples (no hidden layer needed).</summary>
    [Test]
    public async Task TrainSupervised_LinearMap_ConvergesToLowMse()
    {
        // y = 2*x0 - 3*x1 + 0.5
        double[][] inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]];
        double[][] targets = [[2.5], [-2.5], [-0.5], [7.5]];

        var net = DenseNetwork.Create("affine", 2, [], 1, ActivationKind.Linear, new Random(123));
        const int epochs = 4000;
        const double lr = 0.2;
        for (var e = 0; e < epochs; e++)
        {
            for (var i = 0; i < inputs.Length; i++)
                net.TrainSupervised(inputs[i], targets[i], lr);
        }

        var mse = MeanSquaredError(net, inputs, targets);
        Output($"Linear map MSE after {epochs} epochs: {mse:G17}");
        await Assert.That(mse).IsLessThan(1e-3);
    }

    /// <summary>Many cheap updates stay within a wall-clock budget (guards pathological slowdowns).</summary>
    [Test]
    public async Task TrainSupervised_ManySmallUpdates_CompletesWithinWallClockBudget()
    {
        var net = DenseNetwork.Create("throughput", 2, [6], 1, ActivationKind.Tanh, new Random(7));
        double[] input = [0.3, -0.4];
        double[] target = [0.6];
        const int steps = 18_000;
        const double lr = 0.02;

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < steps; i++)
            net.TrainSupervised(input, target, lr);
        sw.Stop();

        Output($"TrainSupervised x{steps} elapsed {sw.ElapsedMilliseconds} ms");
        await Assert.That(sw.ElapsedMilliseconds).IsLessThan(60_000);
    }

    /// <summary>After training, forward predictions move materially toward targets (not only MSE aggregate).</summary>
    [Test]
    public async Task TrainSupervised_SimpleBinarySeparation_ReducesMaxAbsoluteError()
    {
        // Class +1 when x0 > x1, else -1 (linearly separable).
        double[][] inputs = [[0.2, 0.8], [0.9, 0.1], [0.4, 0.45], [0.6, 0.55]];
        double[][] targets = [[-1.0], [1.0], [-1.0], [1.0]];

        var net = DenseNetwork.Create("sep", 2, [5], 1, ActivationKind.Tanh, new Random(11));
        double maxBefore = MaxAbsPredictionError(net, inputs, targets);

        for (var e = 0; e < 2500; e++)
        {
            for (var i = 0; i < inputs.Length; i++)
                net.TrainSupervised(inputs[i], targets[i], 0.12);
        }

        double maxAfter = MaxAbsPredictionError(net, inputs, targets);
        Output($"Max |y-t| before: {maxBefore:G5} after: {maxAfter:G5}");
        await Assert.That(maxAfter).IsLessThan(maxBefore);
        await Assert.That(maxAfter).IsLessThan(0.35);
    }

    private static double MeanSquaredError(DenseNetwork net, double[][] inputs, double[][] targets)
    {
        var sum = 0.0;
        for (var i = 0; i < inputs.Length; i++)
        {
            var o = net.Forward(inputs[i]).ToArray();
            var d = o[0] - targets[i][0];
            sum += d * d;
        }

        return sum / inputs.Length;
    }

    private static double MaxAbsPredictionError(DenseNetwork net, double[][] inputs, double[][] targets)
    {
        var max = 0.0;
        for (var i = 0; i < inputs.Length; i++)
        {
            var o = net.Forward(inputs[i]).ToArray();
            var err = Math.Abs(o[0] - targets[i][0]);
            if (err > max)
                max = err;
        }

        return max;
    }
}
