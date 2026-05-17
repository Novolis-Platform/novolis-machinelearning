using System.Reflection;

namespace Novolis.MachineLearning.TestInfrastructure;

/// <summary>Loads embedded resources using the same dotted naming convention as domain repositories (<see cref="GetManifestName"/>).</summary>
public static class EmbeddedResourceContent
{
    /// <summary>Returns <c>{namespace}.Resources.{fileName}</c> for types colocated with embedded files.</summary>
    public static string GetManifestName(Type anchorType, string fileName)
    {
        ArgumentNullException.ThrowIfNull(anchorType);
        ArgumentException.ThrowIfNullOrEmpty(fileName);
        return $"{anchorType.Namespace}.Resources.{fileName}";
    }

    public static string ReadAllText(Type anchorType, string fileName)
    {
        var assembly = anchorType.Assembly;
        var name = GetManifestName(anchorType, fileName);
        using var stream = assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"Embedded resource not found: {name} in {assembly.GetName().Name}.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    public static Stream OpenRead(Type anchorType, string fileName)
    {
        var assembly = anchorType.Assembly;
        var name = GetManifestName(anchorType, fileName);
        return assembly.GetManifestResourceStream(name)
            ?? throw new InvalidOperationException($"Embedded resource not found: {name} in {assembly.GetName().Name}.");
    }

    public static bool Exists(Type anchorType, string fileName)
    {
        var assembly = anchorType.Assembly;
        var name = GetManifestName(anchorType, fileName);
        return assembly.GetManifestResourceStream(name) is not null;
    }

    public static IReadOnlyList<string> GetManifestResourceNames(Assembly assembly) =>
        assembly.GetManifestResourceNames();
}
