<!-- novolis-pkg-brand:start -->
<p align="center">
  <a href="https://github.com/Novolis-Platform/novolis-machinelearning">
    <img src="https://raw.githubusercontent.com/Novolis-Platform/.github/main/brand/logo-icon.svg" width="72" alt="Novolis"/>
  </a>
</p>
<!-- novolis-pkg-brand:end -->

# Novolis.MachineLearning.Neural

Dense feed-forward networks with supervised training and JSON file persistence.

## Install

```bash
dotnet add package Novolis.MachineLearning.Neural
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Novolis.MachineLearning.Neural;
using Novolis.MachineLearning.Neural.Persistence;

var net = DenseNetwork.Create(
    name: "policy",
    inputSize: 4,
    hiddenSizes: [8],
    outputSize: 2);

double loss = net.TrainSupervised(input, target, learningRate: 0.01);

var repo = new FileNeuralNetworkRepository(
    dataRoot,
    new JsonNeuralNetworkSerializer());
await repo.SaveAsync(net.ToSnapshot(), cancellationToken);
```

Contracts live in `Novolis.MachineLearning.Neural.Abstractions`.

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.MachineLearning.Neural.Abstractions` | Interfaces and snapshots only |
| `Novolis.MachineLearning.Core` | Resolve `NetworksDirectory` paths |

## More documentation

- [Getting started](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/getting-started.md)
- [Design](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/design.md)

## Support

Pre-release (`2026.1.*` on GitHub Packages).

