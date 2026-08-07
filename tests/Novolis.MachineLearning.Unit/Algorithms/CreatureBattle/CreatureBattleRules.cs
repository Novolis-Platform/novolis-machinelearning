namespace Novolis.MachineLearning.Algorithms.Tests.CreatureBattle;

/// <summary>One attack option on a creature card (test oracle only).</summary>
internal readonly record struct AttackOption(int BaseDamage, int EnergyCost, string Name);

/// <summary>Active attacker vs defender board snapshot for KO resolution.</summary>
internal readonly record struct BoardState(
    int BaseDamage,
    int DefenderHp,
    bool HasWeakness,
    bool HasResistance,
    int AttachedEnergy,
    int AttackCost);

/// <summary>Homemade Creature Battle Card Game rules used as a Naive Bayes label oracle.</summary>
internal static class CreatureBattleRules
{
    public const int WeaknessBonus = 20;
    public const int ResistancePenalty = 10;

    public static int EffectiveDamage(int baseDamage, bool hasWeakness, bool hasResistance)
    {
        var damage = baseDamage;
        if (hasWeakness)
            damage += WeaknessBonus;
        if (hasResistance)
            damage -= ResistancePenalty;
        return Math.Max(0, damage);
    }

    public static int EffectiveDamage(in BoardState board) =>
        EffectiveDamage(board.BaseDamage, board.HasWeakness, board.HasResistance);

    public static bool CanPay(int attachedEnergy, int attackCost) =>
        attachedEnergy >= attackCost;

    public static bool CanPay(in BoardState board) =>
        CanPay(board.AttachedEnergy, board.AttackCost);

    /// <summary>KnockOut when the attack can be paid and effective damage covers defender HP.</summary>
    public static CombatOutcome ResolveKnockOut(in BoardState board)
    {
        if (!CanPay(board))
            return CombatOutcome.Survive;

        return EffectiveDamage(board) >= board.DefenderHp
            ? CombatOutcome.KnockOut
            : CombatOutcome.Survive;
    }

    /// <summary>
    /// Among legal attacks (enough energy), pick the highest effective damage.
    /// If none are legal, retreat.
    /// </summary>
    public static AttackChoice ResolveAttackChoice(
        int attachedEnergy,
        bool hasWeakness,
        bool hasResistance,
        AttackOption attackA,
        AttackOption attackB)
    {
        var aLegal = CanPay(attachedEnergy, attackA.EnergyCost);
        var bLegal = CanPay(attachedEnergy, attackB.EnergyCost);

        if (!aLegal && !bLegal)
            return AttackChoice.Retreat;

        if (aLegal && !bLegal)
            return AttackChoice.AttackA;

        if (!aLegal && bLegal)
            return AttackChoice.AttackB;

        var aDmg = EffectiveDamage(attackA.BaseDamage, hasWeakness, hasResistance);
        var bDmg = EffectiveDamage(attackB.BaseDamage, hasWeakness, hasResistance);
        if (aDmg == bDmg)
            return attackA.EnergyCost <= attackB.EnergyCost ? AttackChoice.AttackA : AttackChoice.AttackB;

        return aDmg > bDmg ? AttackChoice.AttackA : AttackChoice.AttackB;
    }
}

internal enum CombatOutcome
{
    Survive,
    KnockOut,
}

internal enum AttackChoice
{
    Retreat,
    AttackA,
    AttackB,
}
