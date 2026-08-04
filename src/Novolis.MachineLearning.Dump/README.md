<!-- novolis-pkg-brand:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.svg" width="72" alt="Novolis"/>
  </a>
</p>
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.Dump

Store trained models as **CodeGen dump** C# fixtures (neural snapshots) and as ML.NET zip pipelines.

## Install

```bash
dotnet add package Novolis.MachineLearning.Dump
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`). Uses `Novolis.CodeGen.Reflection.Dump`.

## Quick start

```csharp
using Novolis.MachineLearning.Dump;
using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;

var net = DenseNetwork.Create("policy", inputSize: 4, hiddenSizes: [8], outputSize: 2);
var repo = new DumpNeuralNetworkRepository(dumpRoot);
await repo.SaveAsync(net.ToSnapshot("best-policy"));
// Writes best-policy.cs (DumpClass fixture) + best-policy.json (runtime load)

var mlStore = new FileMlModelStore(modelsRoot);
mlStore.Save(mlContext, model, schema, "house-regressor");
ITransformer restored = mlStore.Load(mlContext, "house-regressor");
```

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.CodeGen.Reflection.Dump` | Generic object → C# dump without ML types |
| `Novolis.MachineLearning.Neural` | Train dense networks |
| `Novolis.MachineLearning.Algorithms` | Classic ML.NET trainers (Naive Bayes, FastTree, …) |

## Support

Pre-release (`2026.1.*` on GitHub Packages).
