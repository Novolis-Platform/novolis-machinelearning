namespace Novolis.MachineLearning.TestInfrastructure;

/// <summary>Root base with structured <see cref="Output"/>; prefer a layer-specific base (<see cref="UnitTestBase"/>, <see cref="FeatureTestBase"/>, <see cref="IntegrationTestBase"/>) when the test scope is known.</summary>
public abstract class BaseTest
{
    protected void Output<T>(T value) => StructuredTestOutput.WriteObject(value);

    protected void Output(string value) => StructuredTestOutput.WriteLine(value);
}
