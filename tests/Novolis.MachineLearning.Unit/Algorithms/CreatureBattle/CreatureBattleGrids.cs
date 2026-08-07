using Novolis.MachineLearning.Algorithms;
using Novolis.MachineLearning.Algorithms.NaiveBayes;
using Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

namespace Novolis.MachineLearning.Algorithms.Tests;

/// <summary>Enumerates Creature Battle boards for train / holdout splits.</summary>
internal static class CreatureBattleGrids
{
    public static readonly int[] BaseDamages = [20, 30, 40, 50, 60, 70];
    public static readonly int[] DefenderHps = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80];
    public static readonly int[] Energies = [0, 1, 2, 3, 4];
    public static readonly int[] Costs = [1, 2, 3];

    public static IEnumerable<BoardState> AllKnockOutBoards()
    {
        foreach (var bas in BaseDamages)
        foreach (var hp in DefenderHps)
        foreach (var weakness in new[] { false, true })
        foreach (var resistance in new[] { false, true })
        foreach (var energy in Energies)
        foreach (var cost in Costs)
            yield return new BoardState(bas, hp, weakness, resistance, energy, cost);
    }

    public static IEnumerable<BoardState> KnockOutTrainBoards() =>
        AllKnockOutBoards().Where(b => b.DefenderHp % 2 == 0);

    public static IEnumerable<BoardState> KnockOutHoldoutBoards() =>
        AllKnockOutBoards().Where(b => b.DefenderHp % 2 != 0);

    public static IEnumerable<(
        int Energy,
        bool Weakness,
        bool Resistance,
        AttackOption A,
        AttackOption B)> AllAttackChoiceSituations()
    {
        var attacks = new AttackOption[]
        {
            new(20, 1, "EmberSpark"),
            new(40, 2, "TideCrash"),
            new(60, 3, "VerdantSlam"),
            new(35, 1, "EmberFlick"),
            new(55, 2, "TideSurge"),
        };

        foreach (var energy in Energies)
        foreach (var weakness in new[] { false, true })
        foreach (var resistance in new[] { false, true })
        for (var i = 0; i < attacks.Length; i++)
        for (var j = 0; j < attacks.Length; j++)
        {
            if (i == j)
                continue;
            yield return (energy, weakness, resistance, attacks[i], attacks[j]);
        }
    }

    public static IEnumerable<(
        int Energy,
        bool Weakness,
        bool Resistance,
        AttackOption A,
        AttackOption B)> AttackChoiceTrainSituations() =>
        AllAttackChoiceSituations().Where(s => s.Energy % 2 == 0);

    public static IEnumerable<(
        int Energy,
        bool Weakness,
        bool Resistance,
        AttackOption A,
        AttackOption B)> AttackChoiceHoldoutSituations() =>
        AllAttackChoiceSituations().Where(s => s.Energy % 2 != 0);

    public static double Accuracy<TFeature, TLabel>(
        INaiveBayesClassifier<TFeature, TLabel> model,
        IEnumerable<(Features<TFeature> Features, TLabel Label)> rows)
        where TFeature : unmanaged, IEquatable<TFeature>
        where TLabel : notnull
    {
        var total = 0;
        var correct = 0;
        foreach (var (features, label) in rows)
        {
            total++;
            if (EqualityComparer<TLabel>.Default.Equals(model.Predict(features), label))
                correct++;
        }

        return total == 0 ? 0 : correct / (double)total;
    }
}
