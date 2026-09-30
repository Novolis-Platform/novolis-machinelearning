namespace Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

/// <summary>One attack option on a creature card (test oracle only).</summary>
internal readonly record struct AttackOption(int BaseDamage, int EnergyCost, string Name);
