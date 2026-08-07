using Novolis.MachineLearning.Algorithms;
using Novolis.MachineLearning.Algorithms.NaiveBayes;
using Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

namespace Novolis.MachineLearning.Algorithms.Tests;

public sealed class CreatureBattleEdgeCaseTests
{
    [Test]
    public async Task Oracle_ExactDamageEqualsHp_IsKnockOut()
    {
        var board = new BoardState(50, 50, false, false, 2, 2);
        await Assert.That(CreatureBattleRules.EffectiveDamage(board)).IsEqualTo(50);
        await Assert.That(CreatureBattleRules.ResolveKnockOut(board)).IsEqualTo(CombatOutcome.KnockOut);
    }

    [Test]
    public async Task Oracle_OneEnergyShort_SurvivesEvenIfDamageWouldKo()
    {
        var board = new BoardState(80, 40, true, false, 1, 2);
        await Assert.That(CreatureBattleRules.EffectiveDamage(board)).IsGreaterThanOrEqualTo(board.DefenderHp);
        await Assert.That(CreatureBattleRules.ResolveKnockOut(board)).IsEqualTo(CombatOutcome.Survive);
    }

    [Test]
    public async Task Oracle_WeaknessAlone_AddsTwenty()
    {
        await Assert.That(CreatureBattleRules.EffectiveDamage(40, true, false)).IsEqualTo(60);
    }

    [Test]
    public async Task Oracle_ResistanceAlone_SubtractsTen()
    {
        await Assert.That(CreatureBattleRules.EffectiveDamage(40, false, true)).IsEqualTo(30);
    }

    [Test]
    public async Task Oracle_WeaknessAndResistance_NetPlusTen()
    {
        await Assert.That(CreatureBattleRules.EffectiveDamage(40, true, true)).IsEqualTo(50);
    }

    [Test]
    public async Task Oracle_ResistanceCannotDriveDamageBelowZero()
    {
        await Assert.That(CreatureBattleRules.EffectiveDamage(5, false, true)).IsEqualTo(0);
    }

    [Test]
    public async Task Gaussian_ClearCutBoundaryBoards_MatchOracle()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<double, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, CombatOutcome>().Fit(train);

        var boards = new BoardState[]
        {
            new(70, 25, false, false, 3, 1), // deep KO
            new(20, 80, false, true, 3, 1),  // deep Survive
            new(70, 25, false, false, 0, 2), // unpaid Survive
            new(30, 45, true, false, 2, 1),  // 30+20 KO on odd HP holdout band
        };

        foreach (var board in boards)
        {
            var expected = CreatureBattleRules.ResolveKnockOut(board);
            var predicted = model.Predict(CreatureBattleFeatures.ToKnockOutGaussian(board));
            await Assert.That(predicted).IsEqualTo(expected);
        }
    }

    [Test]
    public async Task Bernoulli_InsufficientEnergy_PredictsSurviveOnClearBoards()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Select(b => new LabeledExample<bool, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutBernoulli(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new BernoulliNaiveBayesTrainer<CombatOutcome>().Fit(train);

        var unpaidHeavyHit = new BoardState(70, 20, true, false, 0, 3);
        await Assert.That(CreatureBattleRules.ResolveKnockOut(unpaidHeavyHit)).IsEqualTo(CombatOutcome.Survive);
        await Assert.That(model.Predict(CreatureBattleFeatures.ToKnockOutBernoulli(unpaidHeavyHit)))
            .IsEqualTo(CombatOutcome.Survive);
    }

    [Test]
    public async Task Features_KnockOutEncoders_HaveDocumentedArities()
    {
        var board = new BoardState(40, 50, true, false, 2, 2);
        await Assert.That(CreatureBattleFeatures.ToKnockOutGaussian(board).Length)
            .IsEqualTo(CreatureBattleFeatures.KnockOutGaussianArity);
        await Assert.That(CreatureBattleFeatures.ToKnockOutBernoulli(board).Length)
            .IsEqualTo(CreatureBattleFeatures.KnockOutBernoulliArity);
    }

    [Test]
    public async Task Features_AttackChoiceEncoders_HaveDocumentedArities()
    {
        var a = new AttackOption(20, 1, "EmberSpark");
        var b = new AttackOption(60, 3, "VerdantSlam");
        await Assert.That(CreatureBattleFeatures.ToAttackChoiceGaussian(2, true, false, a, b).Length)
            .IsEqualTo(CreatureBattleFeatures.AttackChoiceGaussianArity);
        await Assert.That(CreatureBattleFeatures.ToAttackChoiceBernoulli(2, true, false, a, b).Length)
            .IsEqualTo(CreatureBattleFeatures.AttackChoiceBernoulliArity);
    }

    [Test]
    public async Task Gaussian_Predict_WrongArity_Throws()
    {
        var train = CreatureBattleGrids.KnockOutTrainBoards()
            .Take(40)
            .Select(b => new LabeledExample<double, CombatOutcome>(
                CreatureBattleFeatures.ToKnockOutGaussian(b),
                CreatureBattleRules.ResolveKnockOut(b)))
            .ToList();

        var model = new GaussianNaiveBayesTrainer<double, CombatOutcome>().Fit(train);
        var act = () => model.Predict(new Features<double>(1, 2, 3));
        await Assert.That(act).Throws<ArgumentException>();
    }

    [Test]
    public async Task Grid_AllBoards_ContainBothOutcomes()
    {
        var outcomes = CreatureBattleGrids.AllKnockOutBoards()
            .Select(b => CreatureBattleRules.ResolveKnockOut(b))
            .Distinct()
            .ToHashSet();

        await Assert.That(outcomes.Contains(CombatOutcome.Survive)).IsTrue();
        await Assert.That(outcomes.Contains(CombatOutcome.KnockOut)).IsTrue();
        await Assert.That(outcomes.Count).IsEqualTo(2);
    }

    [Test]
    public async Task Grid_AttackChoices_ContainRetreatAndBothAttacks()
    {
        var choices = CreatureBattleGrids.AllAttackChoiceSituations()
            .Select(s => CreatureBattleRules.ResolveAttackChoice(s.Energy, s.Weakness, s.Resistance, s.A, s.B))
            .Distinct()
            .ToHashSet();

        await Assert.That(choices.Contains(AttackChoice.Retreat)).IsTrue();
        await Assert.That(choices.Contains(AttackChoice.AttackA)).IsTrue();
        await Assert.That(choices.Contains(AttackChoice.AttackB)).IsTrue();
        await Assert.That(choices.Count).IsEqualTo(3);
    }
}
