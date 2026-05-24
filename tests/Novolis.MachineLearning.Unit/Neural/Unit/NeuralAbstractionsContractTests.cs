using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.TestInfrastructure;

using TUnit.Assertions;

namespace Novolis.MachineLearning.Neural.Tests.Unit;

/// <summary>Ensures the test project references <see cref="Novolis.MachineLearning.Neural.Abstractions"/> types directly.</summary>
public sealed class NeuralAbstractionsContractTests : BaseTest
{
    [Test]
    public async Task ActivationKind_HasExpectedMembers()
    {
        await Assert.That(Enum.GetNames<ActivationKind>().Length).IsGreaterThan(0);
        await Assert.That(Enum.IsDefined(typeof(ActivationKind), ActivationKind.Relu)).IsTrue();
    }
}
