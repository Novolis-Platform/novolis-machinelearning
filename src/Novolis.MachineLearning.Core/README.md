# Novolis.MachineLearning.Core

Repository layout helpers and physical/virtual file system utilities for ML data paths.

## Install

```bash
dotnet add package Novolis.MachineLearning.Core
```

**Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download) (`net10.0`).

## Quick start

```csharp
using Novolis.MachineLearning.Core.IO;
using Novolis.MachineLearning.Core.Paths;

string repoRoot = NovolisFileSystem.RepoRoot;
var fs = NovolisFileSystem.CreatePhysical();
string dataRoot = NovolisMachineLearningDataPaths.ResolveDataRoot(fs);
string networksDir = NovolisMachineLearningDataPaths.NetworksDirectory(fs);
```

Use with `Novolis.MachineLearning.Neural` file repositories for on-disk network snapshots.

## Related packages

| Package | When to use |
|---------|-------------|
| `Novolis.MachineLearning.Neural` | Trainable dense networks |
| `Novolis.MachineLearning.AutoMl` | ML.NET experiment runners |

## More documentation

- [Getting started](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/getting-started.md)
- [Design](https://github.com/Novolis-Platform/novolis-machinelearning/blob/main/docs/design.md)

## Support

Pre-release (`2026.1.*` on GitHub Packages).
