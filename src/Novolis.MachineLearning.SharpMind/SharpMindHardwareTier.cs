namespace Novolis.MachineLearning.SharpMind;

/// <summary>Hardware tier override for SharpMind kernel selection.</summary>
public enum SharpMindHardwareTier
{
    /// <summary>Let SharpMind detect the best available tier.</summary>
    Auto,

    /// <summary>Force FMA kernels.</summary>
    Fma,

    /// <summary>Force AVX2 kernels.</summary>
    Avx2,

    /// <summary>Force SSE kernels.</summary>
    Sse,

    /// <summary>Force scalar kernels.</summary>
    Scalar,
}
