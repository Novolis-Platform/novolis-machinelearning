namespace Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

/// <summary>Active attacker vs defender board snapshot for KO resolution.</summary>
internal readonly record struct BoardState(
    int BaseDamage,
    int DefenderHp,
    bool HasWeakness,
    bool HasResistance,
    int AttachedEnergy,
    int AttackCost);
