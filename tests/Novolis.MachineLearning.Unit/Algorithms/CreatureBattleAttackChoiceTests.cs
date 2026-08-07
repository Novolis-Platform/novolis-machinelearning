using Novolis.MachineLearning.Algorithms.NaiveBayes;
using Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

namespace Novolis.MachineLearning.Algorithms.Tests;

public sealed class CreatureBattleAttackChoiceTests
{
    [Test]
    public async Task Gaussian_HoldoutEnergy_AccuracyAtLeastNinetyPercent()
    {
        var train = CreatureBattleGrids.AttackChoiceTrainSituations()
            .Select(s => new LabeledExample<double, AttackChoice>(
                CreatureBattleFeatures.ToAttackChoiceGaussian(s.Energy, s.Weakness, s.Resistance, s.A, s.B),
                CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, AttackChoice>().Fit(train);

        await Assert.That(model.FeatureCount).IsEqualTo(CreatureBattleFeatures.AttackChoiceGaussianArity);
        await Assert.That(model.Classes.Count).IsEqualTo(3);

        var holdout = CreatureBattleGrids.AttackChoiceHoldoutSituations().ToList();
        await Assert.That(holdout.Count).IsGreaterThan(50);

        var rows = holdout.Select(s => (
            CreatureBattleFeatures.ToAttackChoiceGaussian(s.Energy, s.Weakness, s.Resistance, s.A, s.B),
            CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B)));

        var accuracy = CreatureBattleGrids.Accuracy(model, rows);
        // Raw numeric + legality signals should recover most of the deterministic choice rule.
        await Assert.That(accuracy).IsGreaterThanOrEqualTo(0.90);
    }

    [Test]
    public async Task Bernoulli_HoldoutEnergy_AccuracyAtLeastEightyPercent()
    {
        var train = CreatureBattleGrids.AttackChoiceTrainSituations()
            .Select(s => new LabeledExample<bool, AttackChoice>(
                CreatureBattleFeatures.ToAttackChoiceBernoulli(s.Energy, s.Weakness, s.Resistance, s.A, s.B),
                CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B)))
            .ToList();

        var model = new BernoulliNaiveBayesTrainer<AttackChoice>().Fit(train);

        await Assert.That(model.FeatureCount).IsEqualTo(CreatureBattleFeatures.AttackChoiceBernoulliArity);

        var rows = CreatureBattleGrids.AttackChoiceHoldoutSituations()
            .Select(s => (
                CreatureBattleFeatures.ToAttackChoiceBernoulli(s.Energy, s.Weakness, s.Resistance, s.A, s.B),
                CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B)));

        var accuracy = CreatureBattleGrids.Accuracy(model, rows);
        await Assert.That(accuracy).IsGreaterThanOrEqualTo(0.80);
    }

    [Test]
    public async Task Gaussian_RepresentativeChoices_MatchOracle()
    {
        var train = CreatureBattleGrids.AttackChoiceTrainSituations()
            .Select(s => new LabeledExample<double, AttackChoice>(
                CreatureBattleFeatures.ToAttackChoiceGaussian(s.Energy, s.Weakness, s.Resistance, s.A, s.B),
                CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, AttackChoice>().Fit(train);

        var a = new AttackOption(20, 1, "EmberSpark");
        var b = new AttackOption(60, 3, "VerdantSlam");

        await Assert.That(
                model.Predict(CreatureBattleFeatures.ToAttackChoiceGaussian(0, false, false, a, b)))
            .IsEqualTo(AttackChoice.Retreat);

        await Assert.That(
                model.Predict(CreatureBattleFeatures.ToAttackChoiceGaussian(1, false, false, a, b)))
            .IsEqualTo(AttackChoice.AttackA);

        await Assert.That(
                model.Predict(CreatureBattleFeatures.ToAttackChoiceGaussian(4, false, false, a, b)))
            .IsEqualTo(AttackChoice.AttackB);
    }

    [Test]
    public async Task Oracle_TieBreak_PrefersCheaperAttackWhenDamageEqual()
    {
        var cheap = new AttackOption(40, 1, "EmberFlick");
        var pricey = new AttackOption(40, 3, "TideMirror");

        var choice = CreatureBattleRules.ResolveAttackChoice(
            attachedEnergy: 3,
            hasWeakness: false,
            hasResistance: false,
            attackA: pricey,
            attackB: cheap);

        await Assert.That(choice).IsEqualTo(AttackChoice.AttackB);
    }

    [Test]
    public async Task Bernoulli_RetreatWhenNoEnergy_IsPredicted()
    {
        var train = CreatureBattleGrids.AttackChoiceTrainSituations()
            .Select(s => new LabeledExample<bool, AttackChoice>(
                CreatureBattleFeatures.ToAttackChoiceBernoulli(s.Energy, s.Weakness, s.Resistance, s.A, s.B),
                CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B)))
            .ToList();

        var model = new BernoulliNaiveBayesTrainer<AttackChoice>().Fit(train);
        var a = new AttackOption(40, 2, "TideCrash");
        var b = new AttackOption(60, 3, "VerdantSlam");

        await Assert.That(
                model.Predict(CreatureBattleFeatures.ToAttackChoiceBernoulli(0, false, false, a, b)))
            .IsEqualTo(AttackChoice.Retreat);
    }
}
