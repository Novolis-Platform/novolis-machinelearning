using System.Text.Json;
using System.Text.Json.Serialization;

using TUnit.Core;

namespace Novolis.MachineLearning.TestInfrastructure;

/// <summary>Shared JSON and string output for TUnit <see cref="TestContext"/> (used by <see cref="BaseTest"/> and Playwright layer tests in Novolis.MachineLearning.Playwright.Shared).</summary>
public static class StructuredTestOutput
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        ReferenceHandler = ReferenceHandler.IgnoreCycles
    };

    public static void WriteObject<T>(T value)
    {
        var line = value is null ? "null" : JsonSerializer.Serialize(value, SerializerOptions);
        TestContext.Current?.OutputWriter.WriteLine(line);
    }

    public static void WriteLine(string value) =>
        TestContext.Current?.OutputWriter.WriteLine(value);
}
