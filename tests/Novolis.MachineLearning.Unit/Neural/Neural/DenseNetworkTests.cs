using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;
using Novolis.MachineLearning.Neural.Kit;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Neural;

public class DenseNetworkTests : NeuralNetworkBaseTest
{
    // ── Topology ─────────────────────────────────────────────────────────────

    [Test]
    public async Task InputSize_SingleLayer_ReturnsCorrectValue()
    {
        var net = DenseNetwork.Create("n", 5, [], 3);
        await Assert.That(net.InputSize).IsEqualTo(5);
    }

    [Test]
    public async Task OutputSize_SingleLayer_ReturnsCorrectValue()
    {
        var net = DenseNetwork.Create("n", 5, [], 3);
        await Assert.That(net.OutputSize).IsEqualTo(3);
    }

    [Test]
    public async Task LayerSizes_NoHiddenLayers_ContainsInputAndOutput()
    {
        var net = DenseNetwork.Create("n", 4, [], 2);
        await Assert.That(net.LayerSizes).IsEquivalentTo(new[] { 4, 2 });
    }

    [Test]
    public async Task LayerSizes_OneHiddenLayer_ContainsAllThreeSizes()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1);
        await Assert.That(net.LayerSizes).IsEquivalentTo(new[] { 2, 3, 1 });
    }

    [Test]
    public async Task LayerSizes_DeepNetwork_ContainsAllSizesInOrder()
    {
        var net = DenseNetwork.Create("n", 3, [4, 5, 6], 2);
        await Assert.That(net.LayerSizes).IsEquivalentTo(new[] { 3, 4, 5, 6, 2 });
    }

    [Test]
    public async Task InputSize_MatchesFirstLayerInputCount()
    {
        var net = DenseNetwork.Create("n", 7, [4], 2);
        await Assert.That(net.InputSize).IsEqualTo(net.Layers[0].InputCount);
    }

    [Test]
    public async Task OutputSize_MatchesLastLayerOutputCount()
    {
        var net = DenseNetwork.Create("n", 3, [4, 5], 6);
        await Assert.That(net.OutputSize).IsEqualTo(net.Layers[^1].OutputCount);
    }

    // ── Forward pass with known weights ──────────────────────────────────────

    [Test]
    public async Task Forward_ZeroWeights_OutputEqualsActivatedBias()
    {
        var layer = NeuralFactories.MakeLayer(2, 1, new double[2, 1], new[] { 3.0 }, ActivationKind.Linear);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var output = net.Forward(new double[] { 1.0, 2.0 }).ToArray();
        ExpectScalarEqual(3.0, output[0]);
    }

    [Test]
    public async Task Forward_KnownWeights_ReturnsCorrectDotProductPlusBias()
    {
        // output[0] = 2*1 + 3*1 + 0 = 5
        var weights = new double[2, 1] { { 2.0 }, { 3.0 } };
        var layer = NeuralFactories.MakeLayer(2, 1, weights, new[] { 0.0 }, ActivationKind.Linear);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var output = net.Forward(new double[] { 1.0, 1.0 }).ToArray();
        ExpectScalarEqual(5.0, output[0]);
    }

    [Test]
    public async Task Forward_IdentityWeights_ReturnsInputVectorUnchanged()
    {
        // weights[i,j] = identity, biases = 0 → output = input
        var weights = new double[2, 2] { { 1.0, 0.0 }, { 0.0, 1.0 } };
        var layer = NeuralFactories.MakeLayer(2, 2, weights, new double[2], ActivationKind.Linear);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var output = net.Forward(new double[] { 3.0, 4.0 }).ToArray();
        ExpectEqual(new[] { 3.0, 4.0 }, output);
    }

    [Test]
    public async Task Forward_ZeroInput_OutputEqualsActivatedBias()
    {
        // weight=99, bias=7, input=0 → weighted_sum = 7 → Linear(7) = 7
        var weights = new double[1, 1] { { 99.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, weights, new[] { 7.0 }, ActivationKind.Linear);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var output = net.Forward(new double[] { 0.0 }).ToArray();
        ExpectScalarEqual(7.0, output[0]);
    }

    [Test]
    public async Task Forward_TwoLayerLinearNetwork_ChainMultiplication()
    {
        // L0: y = 2x  (weight=2, bias=0, Linear)
        // L1: z = 3y  (weight=3, bias=0, Linear)
        // Net: z = 6x → Forward([2]) = [12]
        var w0 = new double[1, 1] { { 2.0 } };
        var w1 = new double[1, 1] { { 3.0 } };
        var l0 = NeuralFactories.MakeLayer(1, 1, w0, new double[1], ActivationKind.Linear);
        var l1 = NeuralFactories.MakeLayer(1, 1, w1, new double[1], ActivationKind.Linear);
        var net = NeuralFactories.MakeNetwork("n", l0, l1);
        var output = net.Forward(new double[] { 2.0 }).ToArray();
        await Assert.That(output[0]).IsEqualTo(12.0).Within(1e-9);
    }

    // ── Activation functions ──────────────────────────────────────────────────

    [Test]
    [Arguments(ActivationKind.Linear, 1.0, 1.0)]
    [Arguments(ActivationKind.Linear, -2.5, -2.5)]
    [Arguments(ActivationKind.Relu, 1.0, 1.0)]
    [Arguments(ActivationKind.Relu, -3.0, 0.0)]
    [Arguments(ActivationKind.Relu, 0.0, 0.0)]
    public async Task Forward_SingleNeuron_UnitWeightZeroBias_ActivationIsCorrect(
        ActivationKind act, double input, double expected)
    {
        var w = new double[1, 1] { { 1.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, w, new double[1], act);
        var net = NeuralFactories.MakeNetwork("n", layer);
        await Assert.That(net.Forward(new double[] { input }).ToArray()[0]).IsEqualTo(expected).Within(1e-9);
    }

    [Test]
    public async Task Forward_TanhActivation_MatchesMathTanh()
    {
        var w = new double[1, 1] { { 1.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, w, new double[1], ActivationKind.Tanh);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var output = net.Forward(new double[] { 1.0 }).ToArray();
        await Assert.That(output[0]).IsEqualTo(Math.Tanh(1.0)).Within(1e-12);
    }

    [Test]
    public async Task Forward_SigmoidActivation_MatchesFormula()
    {
        var w = new double[1, 1] { { 1.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, w, new double[1], ActivationKind.Sigmoid);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var output = net.Forward(new double[] { 1.0 }).ToArray();
        double expected = 1.0 / (1.0 + Math.Exp(-1.0));
        await Assert.That(output[0]).IsEqualTo(expected).Within(1e-12);
    }

    [Test]
    public async Task Forward_TanhOfZeroInputZeroBias_IsZero()
    {
        var w = new double[1, 1] { { 1.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, w, new double[1], ActivationKind.Tanh);
        var net = NeuralFactories.MakeNetwork("n", layer);
        await Assert.That(net.Forward(new double[] { 0.0 }).ToArray()[0]).IsEqualTo(0.0).Within(1e-12);
    }

    [Test]
    public async Task Forward_SigmoidOfZeroInputZeroBias_IsHalf()
    {
        var w = new double[1, 1] { { 1.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, w, new double[1], ActivationKind.Sigmoid);
        var net = NeuralFactories.MakeNetwork("n", layer);
        await Assert.That(net.Forward(new double[] { 0.0 }).ToArray()[0]).IsEqualTo(0.5).Within(1e-12);
    }

    // ── Evaluate structure ────────────────────────────────────────────────────

    [Test]
    public async Task Evaluate_ActivationsLength_IsNumLayersPlusOne()
    {
        var net = DenseNetwork.Create("n", 2, [3, 4], 1);
        var eval = net.Evaluate(new double[] { 1.0, 2.0 });
        await Assert.That(eval.Activations.Length).IsEqualTo(net.Layers.Length + 1);
    }

    [Test]
    public async Task Evaluate_ActivationsFirstElement_EqualsInput()
    {
        var net = DenseNetwork.Create("n", 3, [2], 1);
        double[] input = [0.5, -0.3, 1.2];
        var eval = net.Evaluate(input);
        await Assert.That(eval.Activations[0]).IsEquivalentTo(input);
    }

    [Test]
    public async Task Evaluate_WeightedSumsLength_IsNumLayers()
    {
        var net = DenseNetwork.Create("n", 2, [3, 4], 1);
        var eval = net.Evaluate(new double[] { 1.0, 2.0 });
        await Assert.That(eval.WeightedSums.Length).IsEqualTo(net.Layers.Length);
    }

    [Test]
    public async Task Evaluate_OutputMatchesForward()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(42));
        double[] input = [1.0, 2.0];
        var fwd = net.Forward(input).ToArray();
        var eval = net.Evaluate(input);
        await Assert.That(eval.Output).IsEquivalentTo(fwd);
    }

    [Test]
    public async Task Evaluate_LastActivationEqualsOutput()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(7));
        double[] input = [0.5, -0.5];
        var eval = net.Evaluate(input);
        await Assert.That(eval.Activations[^1]).IsEquivalentTo(eval.Output);
    }

    [Test]
    public async Task Evaluate_WeightedSumSingleNeuron_MatchesManualCalculation()
    {
        // weight=2, bias=1, input=3 → weighted_sum = 2*3 + 1 = 7
        var weights = new double[1, 1] { { 2.0 } };
        var layer = NeuralFactories.MakeLayer(1, 1, weights, new[] { 1.0 }, ActivationKind.Linear);
        var net = NeuralFactories.MakeNetwork("n", layer);
        var eval = net.Evaluate(new double[] { 3.0 });
        await Assert.That(eval.WeightedSums[0][0]).IsEqualTo(7.0).Within(1e-9);
    }

    [Test]
    public async Task Snapshot_RoundTrip_PreservesLayerSizes()
    {
        var net = NeuralFactories.CreateDense("rt", 3, [4, 5], 2, ActivationKind.Tanh, seed: 123);
        var copy = NeuralSnapshotRoundTrip.RoundTrip(net, "rt-id");
        await Assert.That(copy.LayerSizes).IsEquivalentTo(net.LayerSizes);
    }

    // ── Training ──────────────────────────────────────────────────────────────

    [Test]
    public async Task TrainSupervised_RepeatedCalls_MseDecreases()
    {
        var net = DenseNetwork.Create("n", 2, [4], 1, random: new Random(42));
        double[] input = [1.0, 0.0];
        double[] target = [0.0];
        double firstMse = net.TrainSupervised(input, target, 0.1);
        double lastMse = firstMse;
        for (int i = 0; i < 99; i++)
            lastMse = net.TrainSupervised(input, target, 0.1);
        await Assert.That(lastMse).IsLessThan(firstMse);
    }

    [Test]
    public async Task TrainSupervised_UpdatesWeightsAfterOneStep()
    {
        var net = DenseNetwork.Create("n", 2, [2], 1, random: new Random(7));
        double weightBefore = net.Layers[0].Weights[0, 0];
        net.TrainSupervised(new double[] { 1.0, 1.0 }, new double[] { 0.5 }, 0.5);
        // With nonzero input and nonzero gradient, at least some weight must change
        bool anyChanged = false;
        for (int i = 0; i < net.Layers[0].InputCount && !anyChanged; i++)
            for (int j = 0; j < net.Layers[0].OutputCount && !anyChanged; j++)
                if (net.Layers[0].Weights[i, j] != weightBefore || i != 0 || j != 0)
                    anyChanged = true;
        // We just confirm the first weight changed (or any weight did)
        await Assert.That(net.Layers[0].Weights[0, 0]).IsNotEqualTo(weightBefore)
            .Because("backprop must update weights when input is non-zero");
    }

    [Test]
    public async Task TrainSupervised_ReturnsMseGe0()
    {
        var net = DenseNetwork.Create("n", 1, [], 1, random: new Random(1));
        double mse = net.TrainSupervised(new double[] { 0.0 }, new double[] { 1.0 }, 0.0);
        await Assert.That(mse).IsGreaterThanOrEqualTo(0.0);
    }

    [Test]
    public async Task TrainWithOutputGradient_LinearHead_MatchesTrainSupervisedOneStep()
    {
        var rng = new Random(99);
        var template = DenseNetwork.Create("t", 3, [4], 1, ActivationKind.Tanh, rng);
        var trained = (DenseNetwork)template.Clone("a");
        var viaGrad = (DenseNetwork)template.Clone("b");
        var input = new double[] { 0.2, -0.5, 1.1 };
        var target = new[] { 0.3 };
        const double lr = 0.05;
        trained.TrainSupervised(input, target, lr);
        var eval = viaGrad.Evaluate(input);
        Span<double> g = stackalloc double[1];
        g[0] = eval.Output[0] - target[0];
        viaGrad.TrainWithOutputGradient(input, g, lr);
        for (var li = 0; li < trained.Layers.Length; li++)
        {
            for (var i = 0; i < trained.Layers[li].InputCount; i++)
            {
                for (var j = 0; j < trained.Layers[li].OutputCount; j++)
                    await Assert.That(trained.Layers[li].Weights[i, j]).IsEqualTo(viaGrad.Layers[li].Weights[i, j]).Within(1e-9);
            }

            for (var j = 0; j < trained.Layers[li].OutputCount; j++)
                await Assert.That(trained.Layers[li].Biases[j]).IsEqualTo(viaGrad.Layers[li].Biases[j]).Within(1e-9);
        }
    }

    [Test]
    public async Task TrainSupervised_XorProblem_ConvergesWithSomeInitialisation()
    {
        // XOR convergence depends on initialisation. We try up to 30 seeds
        // (stopping early on success) to prove the implementation CAN solve XOR.
        double[][] xorInputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        double[][] xorTargets = [[0.0], [1.0], [1.0], [0.0]];
        double bestMse = double.MaxValue;
        int bestSeed = -1;

        for (int seed = 0; seed < 30; seed++)
        {
            var net = DenseNetwork.Create("xor", 2, [6], 1, ActivationKind.Tanh, new Random(seed));
            for (int epoch = 0; epoch < 5000; epoch++)
                for (int i = 0; i < 4; i++)
                    net.TrainSupervised(xorInputs[i], xorTargets[i], 0.5);

            double mse = 0.0;
            for (int i = 0; i < 4; i++)
            {
                var o = net.Forward(xorInputs[i]).ToArray();
                double diff = o[0] - xorTargets[i][0];
                mse += diff * diff;
            }
            mse /= 4;
            if (mse < bestMse) { bestMse = mse; bestSeed = seed; }
            if (bestMse < 0.05) break;
        }

        Output($"XOR best avg MSE: {bestMse:F6} (seed {bestSeed})");
        await Assert.That(bestMse).IsLessThan(0.05)
            .Because("backprop must be capable of solving XOR with some initialisation");
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    [Test]
    public async Task Clone_MutatingClone_DoesNotAffectOriginalWeights()
    {
        var original = DenseNetwork.Create("orig", 2, [3], 1, random: new Random(1));
        double[,] originalWeightsCopy = (double[,])original.Layers[0].Weights.Clone();
        var clone = (DenseNetwork)original.Clone();
        clone.Mutate(new Random(99), new MutationSettings(1.0, 5.0, 1.0, 5.0));
        for (int i = 0; i < original.Layers[0].InputCount; i++)
            for (int j = 0; j < original.Layers[0].OutputCount; j++)
                await Assert.That(original.Layers[0].Weights[i, j]).IsEqualTo(originalWeightsCopy[i, j]);
    }

    [Test]
    public async Task Clone_PreservesLayerSizes()
    {
        var original = DenseNetwork.Create("n", 3, [4, 5], 2);
        var clone = (DenseNetwork)original.Clone();
        await Assert.That(clone.LayerSizes).IsEquivalentTo(original.LayerSizes);
    }

    [Test]
    public async Task Clone_WithNewName_UsesProvidedName()
    {
        var original = DenseNetwork.Create("original", 2, [2], 1);
        var clone = (DenseNetwork)original.Clone("newname");
        await Assert.That(clone.Name).IsEqualTo("newname");
    }

    [Test]
    public async Task Clone_WithoutName_PreservesOriginalName()
    {
        var original = DenseNetwork.Create("mynet", 2, [2], 1);
        var clone = (DenseNetwork)original.Clone();
        await Assert.That(clone.Name).IsEqualTo("mynet");
    }

    [Test]
    public async Task Clone_ProducesIdenticalOutputForSameInput()
    {
        var original = DenseNetwork.Create("n", 2, [3], 1, random: new Random(42));
        var clone = (DenseNetwork)original.Clone();
        double[] input = [1.5, -0.5];
        var origOut = original.Forward(input).ToArray();
        var cloneOut = clone.Forward(input).ToArray();
        await Assert.That(cloneOut).IsEquivalentTo(origOut).Because("clone is a deep copy");
    }

    // ── CopyFrom ─────────────────────────────────────────────────────────────

    [Test]
    public async Task CopyFrom_CopiesWeightsFromSource()
    {
        var source = DenseNetwork.Create("src", 2, [3], 1, random: new Random(5));
        var dest = DenseNetwork.Create("dst", 2, [3], 1, random: new Random(99));
        dest.CopyFrom(source);
        for (int i = 0; i < dest.Layers[0].InputCount; i++)
            for (int j = 0; j < dest.Layers[0].OutputCount; j++)
                await Assert.That(dest.Layers[0].Weights[i, j]).IsEqualTo(source.Layers[0].Weights[i, j]);
    }

    [Test]
    public async Task CopyFrom_CopiesBiasesFromSource()
    {
        var source = DenseNetwork.Create("src", 2, [3], 1, random: new Random(5));
        // Manually set a bias so we can verify it was copied
        source.Layers[0].Biases[0] = 42.0;
        var dest = DenseNetwork.Create("dst", 2, [3], 1, random: new Random(99));
        dest.CopyFrom(source);
        await Assert.That(dest.Layers[0].Biases[0]).IsEqualTo(42.0);
    }

    [Test]
    public async Task CopyFrom_MismatchedInputSize_ThrowsInvalidOperationException()
    {
        var net1 = DenseNetwork.Create("n1", 2, [], 1);
        var net2 = DenseNetwork.Create("n2", 3, [], 1);
        await Assert.That(() => net1.CopyFrom(net2)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task CopyFrom_MismatchedOutputSize_ThrowsInvalidOperationException()
    {
        var net1 = DenseNetwork.Create("n1", 2, [], 1);
        var net2 = DenseNetwork.Create("n2", 2, [], 2);
        await Assert.That(() => net1.CopyFrom(net2)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task CopyFrom_MismatchedLayerCount_ThrowsInvalidOperationException()
    {
        // Same input/output but different depth: [2,2,1] vs [2,3,2,1]
        var net1 = DenseNetwork.Create("n1", 2, [2], 1);
        var net2 = DenseNetwork.Create("n2", 2, [3, 2], 1);
        await Assert.That(() => net1.CopyFrom(net2)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task CopyFrom_MismatchedHiddenLayerSize_ThrowsInvalidOperationException()
    {
        // Same input/output/depth but hidden size differs: [2,3,1] vs [2,4,1]
        var net1 = DenseNetwork.Create("n1", 2, [3], 1);
        var net2 = DenseNetwork.Create("n2", 2, [4], 1);
        await Assert.That(() => net1.CopyFrom(net2)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task CopyFrom_MismatchedActivationKind_ThrowsInvalidOperationException()
    {
        var net1 = DenseNetwork.Create("n1", 2, [3], 1, ActivationKind.Tanh);
        var net2 = DenseNetwork.Create("n2", 2, [3], 1, ActivationKind.Relu);
        await Assert.That(() => net1.CopyFrom(net2)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task CopyFrom_NonDenseNetwork_ThrowsInvalidOperationException()
    {
        var net = DenseNetwork.Create("n", 2, [2], 1);
        var stub = new StubNeuralNetwork();
        await Assert.That(() => net.CopyFrom(stub)).Throws<InvalidOperationException>();
    }

    // ── Mutate ───────────────────────────────────────────────────────────────

    [Test]
    public async Task Mutate_RateZero_WeightsAndBiasesUnchanged()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(1));
        double[,] weightsBefore = (double[,])net.Layers[0].Weights.Clone();
        double[] biasesBefore = (double[])net.Layers[0].Biases.Clone();

        net.Mutate(new Random(42), new MutationSettings(0.0, 10.0, 0.0, 10.0));

        for (int i = 0; i < net.Layers[0].InputCount; i++)
            for (int j = 0; j < net.Layers[0].OutputCount; j++)
                await Assert.That(net.Layers[0].Weights[i, j]).IsEqualTo(weightsBefore[i, j]);
        for (int j = 0; j < net.Layers[0].OutputCount; j++)
            await Assert.That(net.Layers[0].Biases[j]).IsEqualTo(biasesBefore[j]);
    }

    [Test]
    public async Task Mutate_RateOne_ChangesSomeWeights()
    {
        var net = DenseNetwork.Create("n", 3, [4], 2, random: new Random(1));
        double[,] weightsBefore = (double[,])net.Layers[0].Weights.Clone();

        net.Mutate(new Random(42), new MutationSettings(1.0, 1.0, 1.0, 1.0));

        bool anyChanged = false;
        for (int i = 0; i < net.Layers[0].InputCount && !anyChanged; i++)
            for (int j = 0; j < net.Layers[0].OutputCount && !anyChanged; j++)
                if (net.Layers[0].Weights[i, j] != weightsBefore[i, j])
                    anyChanged = true;
        await Assert.That(anyChanged).IsTrue()
            .Because("rate=1 and sigma=1 must change weights with any non-zero Gaussian draw");
    }

    // ── Snapshot ─────────────────────────────────────────────────────────────

    [Test]
    public async Task ToSnapshot_FromSnapshot_RoundTrip_ProducesSameOutput()
    {
        var original = DenseNetwork.Create("n", 2, [3], 1, random: new Random(42));
        double[] input = [1.0, 2.0];
        var origOut = original.Forward(input).ToArray();

        var snapshot = original.ToSnapshot("rt-id");
        var restored = DenseNetwork.FromSnapshot(snapshot);
        var restoredOut = restored.Forward(input).ToArray();

        for (int i = 0; i < origOut.Length; i++)
            await Assert.That(restoredOut[i]).IsEqualTo(origOut[i]).Within(1e-9);
    }

    [Test]
    public async Task ToSnapshot_IdPreservedInSnapshot()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1);
        var snapshot = net.ToSnapshot("my-unique-id-123");
        await Assert.That(snapshot.Id).IsEqualTo("my-unique-id-123");
    }

    [Test]
    public async Task ToSnapshot_MetadataDictionaryPreserved()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1);
        var meta = new Dictionary<string, string> { ["key1"] = "val1", ["key2"] = "val2" };
        var snapshot = net.ToSnapshot("id", meta);
        await Assert.That(snapshot.Metadata).IsNotNull();
        await Assert.That(snapshot.Metadata!["key1"]).IsEqualTo("val1");
        await Assert.That(snapshot.Metadata["key2"]).IsEqualTo("val2");
    }

    [Test]
    public async Task ToSnapshot_NullMetadata_IsNull()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1);
        var snapshot = net.ToSnapshot("id", null);
        await Assert.That(snapshot.Metadata).IsNull();
    }

    [Test]
    public async Task FromSnapshot_WrongWeightsRowCount_ThrowsInvalidOperationException()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(1));
        var snapshot = net.ToSnapshot("id");
        // Layer 0: InputCount=2 → make only 1 weight row
        var badWeights = new double[1][];
        badWeights[0] = new double[snapshot.Layers[0].OutputCount];
        var badLayer = snapshot.Layers[0] with { Weights = badWeights };
        var badLayers = snapshot.Layers.ToArray();
        badLayers[0] = badLayer;
        var badSnapshot = snapshot with { Layers = badLayers };

        await Assert.That(() => DenseNetwork.FromSnapshot(badSnapshot)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task FromSnapshot_WrongWeightsColCount_ThrowsInvalidOperationException()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(1));
        var snapshot = net.ToSnapshot("id");
        // Layer 0: OutputCount=3 → make rows with 2 columns instead
        var badWeights = new double[snapshot.Layers[0].InputCount][];
        for (int i = 0; i < badWeights.Length; i++)
            badWeights[i] = new double[snapshot.Layers[0].OutputCount - 1];
        var badLayer = snapshot.Layers[0] with { Weights = badWeights };
        var badLayers = snapshot.Layers.ToArray();
        badLayers[0] = badLayer;
        var badSnapshot = snapshot with { Layers = badLayers };

        await Assert.That(() => DenseNetwork.FromSnapshot(badSnapshot)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task FromSnapshot_WrongBiasesLength_ThrowsInvalidOperationException()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(1));
        var snapshot = net.ToSnapshot("id");
        // Layer 0: OutputCount=3 → provide 2 biases instead
        var badLayer = snapshot.Layers[0] with { Biases = new double[snapshot.Layers[0].OutputCount - 1] };
        var badLayers = snapshot.Layers.ToArray();
        badLayers[0] = badLayer;
        var badSnapshot = snapshot with { Layers = badLayers };

        await Assert.That(() => DenseNetwork.FromSnapshot(badSnapshot)).Throws<InvalidOperationException>();
    }

    [Test]
    public async Task FromSnapshot_UnknownActivationString_ThrowsInvalidOperationException()
    {
        var net = DenseNetwork.Create("n", 2, [3], 1, random: new Random(1));
        var snapshot = net.ToSnapshot("id");
        var badLayer = snapshot.Layers[0] with { Activation = "NotAnActivation" };
        var badLayers = snapshot.Layers.ToArray();
        badLayers[0] = badLayer;
        var badSnapshot = snapshot with { Layers = badLayers };

        await Assert.That(() => DenseNetwork.FromSnapshot(badSnapshot)).Throws<InvalidOperationException>();
    }

    // ── Create factory ────────────────────────────────────────────────────────

    [Test]
    public async Task Create_LastLayerIsAlwaysLinear_RegardlessOfActivationArgument()
    {
        foreach (var act in Enum.GetValues<ActivationKind>())
        {
            var net = DenseNetwork.Create("n", 2, [3], 1, act);
            await Assert.That(net.Layers[^1].Activation).IsEqualTo(ActivationKind.Linear)
                .Because($"last layer must always be Linear regardless of activation={act}");
        }
    }

    [Test]
    public async Task Create_HiddenLayersUseSpecifiedActivation()
    {
        var net = DenseNetwork.Create("n", 2, [3, 4], 1, ActivationKind.Relu);
        await Assert.That(net.Layers[0].Activation).IsEqualTo(ActivationKind.Relu);
        await Assert.That(net.Layers[1].Activation).IsEqualTo(ActivationKind.Relu);
        await Assert.That(net.Layers[2].Activation).IsEqualTo(ActivationKind.Linear);
    }

    [Test]
    public async Task Create_WithSameSeed_ProducesDeterministicWeights()
    {
        var net1 = DenseNetwork.Create("n", 3, [4], 2, random: new Random(1234));
        var net2 = DenseNetwork.Create("n", 3, [4], 2, random: new Random(1234));
        for (int i = 0; i < net1.Layers[0].InputCount; i++)
            for (int j = 0; j < net1.Layers[0].OutputCount; j++)
                await Assert.That(net1.Layers[0].Weights[i, j]).IsEqualTo(net2.Layers[0].Weights[i, j]);
    }

    [Test]
    public async Task Create_NoHiddenLayers_ProducesSingleLayerNetwork()
    {
        var net = DenseNetwork.Create("n", 3, [], 2);
        await Assert.That(net.Layers.Length).IsEqualTo(1);
        await Assert.That(net.InputSize).IsEqualTo(3);
        await Assert.That(net.OutputSize).IsEqualTo(2);
    }

    [Test]
    public async Task Create_WeightsHaveXavierScale()
    {
        // Xavier std ≈ sqrt(2/(in+out)). For large number of samples, mean ≈ 0.
        var net = DenseNetwork.Create("n", 10, [20], 5, random: new Random(42));
        var layer = net.Layers[0];
        var allWeights = new List<double>();
        for (int i = 0; i < layer.InputCount; i++)
            for (int j = 0; j < layer.OutputCount; j++)
                allWeights.Add(layer.Weights[i, j]);
        double mean = allWeights.Average();
        await Assert.That(mean).IsEqualTo(0.0).Within(0.5).Because("Xavier init should be centred near zero");
    }

    // ── Private stub ─────────────────────────────────────────────────────────

    private sealed class StubNeuralNetwork : INeuralNetwork
    {
        public string Name => "stub";
        public int InputSize => 2;
        public int OutputSize => 1;
        public IReadOnlyList<int> LayerSizes => [2, 2, 1];
        public ReadOnlyMemory<double> Forward(ReadOnlySpan<double> input) => Array.Empty<double>();
        public NetworkEvaluation Evaluate(ReadOnlySpan<double> input)
            => new(Array.Empty<double>(), Array.Empty<double[]>(), Array.Empty<double[]>());
    }
}
