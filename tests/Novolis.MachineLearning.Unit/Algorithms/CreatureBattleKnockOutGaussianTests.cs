using Novolis.MachineLearning.Algorithms;
using Novolis.MachineLearning.Algorithms.NaiveBayes;
using Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

namespace Novolis.MachineLearning.Algorithms.Tests;

public sealed class CreatureBattleKnockOutGaussianTests
{
    [Test]
    public async Task Fit_OnEvenHpGrid_HoldoutOddHp_AccuracyAtLeastNinetyPercent()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<double, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, CombatOutcome>().Fit(train);

        await Assert.That(model.FeatureCount).IsEqualTo(CreatureBattleFeatures.KnockOutGaussianArity);
        await Assert.That(model.Classes.Count).IsEqualTo(2);

        var holdout = CreatureBattleGrids.KnockOutHoldoutBoards().ToList();
        await Assert.That(holdout.Count).IsGreaterThan(100);

        var rows = holdout.Select(b => (
            CreatureBattleFeatures.ToKnockOutGaussian(b),
            CreatureBattleRules.ResolveKnockOut(b)));

        var accuracy = CreatureBattleGrids.Accuracy(model, rows);
        // Naive Bayes independence assumption cannot perfectly encode (damageGap AND energySurplus).
        await Assert.That(accuracy).IsGreaterThanOrEqualTo(0.90);
    }

    [Test]
    public async Task Predict_ClearCutBoards_MatchOracle()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<double, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, CombatOutcome>().Fit(train);

        var cases = new (BoardState Board, CombatOutcome Expected)[]
        {
            (new(70, 25, false, false, 3, 1), CombatOutcome.KnockOut),
            (new(20, 80, false, false, 3, 1), CombatOutcome.Survive),
            (new(70, 25, true, false, 0, 2), CombatOutcome.Survive),  // unpaid
            (new(40, 55, true, false, 2, 1), CombatOutcome.KnockOut), // 40+20
            (new(50, 55, false, true, 3, 1), CombatOutcome.Survive), // 50-10
        };

        foreach (var (board, expected) in cases)
        {
            await Assert.That(CreatureBattleRules.ResolveKnockOut(board)).IsEqualTo(expected);
            var predicted = model.Predict(CreatureBattleFeatures.ToKnockOutGaussian(board));
            await Assert.That(predicted).IsEqualTo(expected);
        }
    }

    [Test]
    public async Task PredictScores_OnCreatureBoard_ProbabilitiesSumToOne()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<double, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, CombatOutcome>().Fit(train);
        var board = new BoardState(55, 45, true, false, 2, 2);
        var scores = model.PredictScores(CreatureBattleFeatures.ToKnockOutGaussian(board));

        await Assert.That(scores.Count).IsEqualTo(2);
        await Assert.That(scores.Sum(s => s.Probability)).IsEqualTo(1.0).Within(1e-9);

        var top = scores.OrderByDescending(s => s.Probability).First();
        await Assert.That(top.Label).IsEqualTo(model.Predict(CreatureBattleFeatures.ToKnockOutGaussian(board)));
    }

    [Test]
    public async Task HoldoutAccuracy_BeatsMajorityBaseline()
    {
        var holdout = CreatureBattleGrids.KnockOutHoldoutBoards().ToList();
        var knockOutShare = holdout.Count(b => CreatureBattleRules.ResolveKnockOut(b) == CombatOutcome.KnockOut)
            / (double)holdout.Count;
        var majority = Math.Max(knockOutShare, 1.0 - knockOutShare);

        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<double, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, CombatOutcome>().Fit(train);
        var accuracy = CreatureBattleGrids.Accuracy(
            model,
            holdout.Select(b => (
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b))));

        await Assert.That(accuracy).IsGreaterThan(majority + 0.15);
        await Assert.That(accuracy).IsGreaterThanOrEqualTo(0.90);
    }
}
