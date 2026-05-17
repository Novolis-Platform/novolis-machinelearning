namespace Novolis.MachineLearning.Core.Tests.Paths;

using Novolis.MachineLearning.Core.Paths;

using TUnit.Assertions;

public sealed class NovolisMachineLearningRepoPathsTests
{
    [Test]
    public async Task TryGetRepoRoot_FindsMarker_WalkingUpFromNestedDirectory()
    {
        var tmp = Path.Combine(Path.GetTempPath(), "NovolisMachineLearningRepoPathsTests-" + Guid.NewGuid().ToString("N"));
        var nested = Path.Combine(tmp, "artifacts", "bin", "App", "debug");
        try
        {
            Directory.CreateDirectory(nested);
            await File.WriteAllTextAsync(Path.Combine(tmp, "Novolis.MachineLearning.slnx"), "<Solution />");

            var root = NovolisMachineLearningRepoPaths.TryGetRepoRoot(nested);
            await Assert.That(root).IsNotNull();
            await Assert.That(Path.GetFullPath(root!)).IsEqualTo(Path.GetFullPath(tmp));
        }
        finally
        {
            try
            {
                Directory.Delete(tmp, recursive: true);
            }
            catch
            {
                // temp cleanup best-effort
            }
        }
    }

    [Test]
    public async Task NeuralLabsRunnerSessionsRoot_ContainsNeuralLabsRunnerSegment()
    {
        var s = NovolisMachineLearningRepoPaths.NeuralLabsRunnerSessionsRoot();
        await Assert.That(Path.IsPathRooted(s)).IsTrue();
        var n = s.Replace('/', Path.DirectorySeparatorChar);
        await Assert.That(n).Contains($"{Path.DirectorySeparatorChar}neural-labs{Path.DirectorySeparatorChar}runner");
    }
}
