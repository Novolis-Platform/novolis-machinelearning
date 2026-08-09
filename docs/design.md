# Design

Machine learning core, AutoML, and neural utilities for Novolis.

Published docs: [https://novolis-platform.github.io/.github/novolis-machinelearning/](https://novolis-platform.github.io/.github/novolis-machinelearning/)

## Layer placement

Follow [library-boundaries](https://github.com/Novolis-Platform/novolis-governance/blob/main/docs/library-boundaries.md) for layer placement.

## Goals

- Keep public APIs documented and packable as `Novolis.*` on GitHub Packages (when applicable).
- Prefer BCL types and existing Novolis packages over parallel abstractions.
- Document restore and ProjectReference-mode builds without local NuGet folder feeds.

## Non-goals

- Local NuGet folder feeds or committed cross-repo `ProjectReference` into sibling checkouts.
- Avalonia package references outside `Novolis.Avalonia.*`.
- Upward spine dependencies (e.g. Math → Simulation).

## Packages

- `Novolis.MachineLearning.Algorithms`
- `Novolis.MachineLearning.AutoMl`
- `Novolis.MachineLearning.Core`
- `Novolis.MachineLearning.Dump`
- `Novolis.MachineLearning.Neural`
- `Novolis.MachineLearning.Neural.Abstractions`

## Topics

- `dotnet`
- `machine-learning`
- `novolis`
