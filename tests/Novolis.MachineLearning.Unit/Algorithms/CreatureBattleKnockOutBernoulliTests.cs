using Novolis.MachineLearning.Algorithms.NaiveBayes;
using Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

namespace Novolis.MachineLearning.Algorithms.Tests;

public sealed class CreatureBattleKnockOutBernoulliTests
{
    [Test]
    public async Task Fit_OnEvenHpGrid_HoldoutAccuracyAtLeastNinetyPercent()
    {
        // Threshold features are lossy by design — floor documents that, not "duel winner".
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<bool, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutBernoulli(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new BernoulliNaiveBayesTrainer<CombatOutcome>().Fit(train);

        await Assert.That(model.FeatureCount).IsEqualTo(CreatureBattleFeatures.KnockOutBernoulliArity);
        await Assert.That(model.Classes.Count).IsEqualTo(2);

        var holdout = CreatureBattleGrids.KnockOutHoldoutBoards().ToList();
        await Assert.That(holdout.Count).IsGreaterThan(100);

        var rows = holdout.Select(b => (
            CreatureBattleFeatures.ToKnockOutBernoulli(b),
            CreatureBattleRules.ResolveKnockOut(b)));

        var accuracy = CreatureBattleGrids.Accuracy(model, rows);
        await Assert.That(accuracy).IsGreaterThanOrEqualTo(0.90);
    }

    [Test]
    public async Task Predict_ClearCutBoards_MatchOracle()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<bool, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutBernoulli(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new BernoulliNaiveBayesTrainer<CombatOutcome>().Fit(train);

        // Strong signals for the threshold encoding.
        var knockOut = new BoardState(70, 30, true, false, 3, 1);
        var surviveUnpaid = new BoardState(70, 30, true, false, 0, 2);
        var surviveLowDamage = new BoardState(20, 80, false, true, 3, 1);

        await Assert.That(model.Predict(CreatureBattleFeatures.ToKnockOutBernoulli(knockOut)))
            .IsEqualTo(CombatOutcome.KnockOut);
        await Assert.That(model.Predict(CreatureBattleFeatures.ToKnockOutBernoulli(surviveUnpaid)))
            .IsEqualTo(CombatOutcome.Survive);
        await Assert.That(model.Predict(CreatureBattleFeatures.ToKnockOutBernoulli(surviveLowDamage)))
            .IsEqualTo(CombatOutcome.Survive);
    }

    [Test]
    public async Task PredictScores_ProbabilitiesSumToOne()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<bool, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutBernoulli(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new BernoulliNaiveBayesTrainer<CombatOutcome>().Fit(train);
        var scores = model.PredictScores(
            CreatureBattleFeatures.ToKnockOutBernoulli(new BoardState(50, 45, false, false, 2, 2)));

        await Assert.That(scores.Sum(s => s.Probability)).IsEqualTo(1.0).Within(1e-9);
    }

    [Test]
    public async Task TrainSet_IsLargerThanHoldout_AndBothNonEmpty()
    {
        var trainCount = CreatureBattleGrids.KnockOutTrainBoards().Count();
        var holdoutCount = CreatureBattleGrids.KnockOutHoldoutBoards().Count();
        await Assert.That(trainCount).IsGreaterThan(0);
        await Assert.That(holdoutCount).IsGreaterThan(0);
        await Assert.That(trainCount).IsGreaterThan(holdoutCount);
    }
}
