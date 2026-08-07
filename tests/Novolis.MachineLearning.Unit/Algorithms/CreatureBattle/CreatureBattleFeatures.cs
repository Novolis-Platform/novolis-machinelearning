using Novolis.MachineLearning.Algorithms;

namespace Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

/// <summary>Honest feature encoders — no single outcome / KO bit.</summary>
internal static class CreatureBattleFeatures
{
    public const int KnockOutGaussianArity = 9;
    public const int KnockOutBernoulliArity = 7;
    public const int AttackChoiceGaussianArity = 10;
    public const int AttackChoiceBernoulliArity = 8;

    /// <summary>
    /// Raw board fields plus derived continuous signals (effective damage, energy surplus, damage gap).
    /// Does not include the KnockOut boolean itself.
    /// </summary>
    public static Features<double> ToKnockOutGaussian(in BoardState board)
    {
        var effective = CreatureBattleRules.EffectiveDamage(board);
        return new(
            board.BaseDamage,
            board.DefenderHp,
            board.HasWeakness ? 1.0 : 0.0,
            board.HasResistance ? 1.0 : 0.0,
            board.AttachedEnergy,
            board.AttackCost,
            effective,
            board.AttachedEnergy - board.AttackCost,
            effective - board.DefenderHp);
    }

    /// <summary>
    /// Threshold / flag bits only. Includes partial damage-vs-HP probes under weakness/resistance
    /// scenarios, but never a combined KnockOut label bit.
    /// </summary>
    public static Features<bool> ToKnockOutBernoulli(in BoardState board) =>
        new(
            board.BaseDamage >= 50,
            board.DefenderHp <= 40,
            board.HasWeakness,
            board.HasResistance,
            board.AttachedEnergy >= board.AttackCost,
            board.BaseDamage + CreatureBattleRules.WeaknessBonus >= board.DefenderHp,
            board.BaseDamage - CreatureBattleRules.ResistancePenalty >= board.DefenderHp);

    public static Features<double> ToAttackChoiceGaussian(
        int attachedEnergy,
        bool hasWeakness,
        bool hasResistance,
        AttackOption attackA,
        AttackOption attackB)
    {
        var dmgA = CreatureBattleRules.EffectiveDamage(attackA.BaseDamage, hasWeakness, hasResistance);
        var dmgB = CreatureBattleRules.EffectiveDamage(attackB.BaseDamage, hasWeakness, hasResistance);
        var legalA = attachedEnergy >= attackA.EnergyCost ? 1.0 : 0.0;
        var legalB = attachedEnergy >= attackB.EnergyCost ? 1.0 : 0.0;
        return new(
            attachedEnergy,
            hasWeakness ? 1.0 : 0.0,
            hasResistance ? 1.0 : 0.0,
            attackA.EnergyCost,
            attackB.EnergyCost,
            dmgA,
            dmgB,
            dmgA - dmgB,
            legalA,
            legalB);
    }

    public static Features<bool> ToAttackChoiceBernoulli(
        int attachedEnergy,
        bool hasWeakness,
        bool hasResistance,
        AttackOption attackA,
        AttackOption attackB)
    {
        var dmgA = CreatureBattleRules.EffectiveDamage(attackA.BaseDamage, hasWeakness, hasResistance);
        var dmgB = CreatureBattleRules.EffectiveDamage(attackB.BaseDamage, hasWeakness, hasResistance);
        return new(
            attachedEnergy >= attackA.EnergyCost,
            attachedEnergy >= attackB.EnergyCost,
            hasWeakness,
            hasResistance,
            dmgA > dmgB,
            dmgA >= 40,
            dmgB >= 40,
            attackA.EnergyCost <= attackB.EnergyCost);
    }
}
