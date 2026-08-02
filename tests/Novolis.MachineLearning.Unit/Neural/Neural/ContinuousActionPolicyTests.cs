using Novolis.MachineLearning.Neural;

namespace Novolis.MachineLearning.Unit.Neural;

public class ContinuousActionPolicyTests
{
    [Test]
    public async Task Create_Act_ReturnsClampedActions()
    {
        var policy = ContinuousActionPolicy.Create(
            "arcade",
            observationSize: 4,
            actionSize: 3,
            hiddenSizes: [8],
            random: new Random(1));

        var obs = new double[] { 0.5, -0.25, 0.1, 0.0 };
        var actions = new double[3];
        policy.Act(obs, actions);

        await Assert.That(actions[0]).IsGreaterThanOrEqualTo(-1.0);
        await Assert.That(actions[0]).IsLessThanOrEqualTo(1.0);
        await Assert.That(actions[1]).IsGreaterThanOrEqualTo(-1.0);
        await Assert.That(actions[2]).IsLessThanOrEqualTo(1.0);
    }

    [Test]
    public async Task Imitate_ReducesErrorTowardTarget()
    {
        var policy = ContinuousActionPolicy.Create(
            "imitate",
            observationSize: 2,
            actionSize: 2,
            hiddenSizes: [12],
            random: new Random(7));

        var obs = new double[] { 0.4, -0.2 };
        var target = new double[] { 0.8, -0.5 };
        var actions = new double[2];

        policy.Act(obs, actions);
        var before = Math.Abs(actions[0] - target[0]) + Math.Abs(actions[1] - target[1]);

        for (var i = 0; i < 80; i++)
            _ = policy.Imitate(obs, target, learningRate: 0.08);

        policy.Act(obs, actions);
        var after = Math.Abs(actions[0] - target[0]) + Math.Abs(actions[1] - target[1]);
        await Assert.That(after).IsLessThan(before);
    }
}
