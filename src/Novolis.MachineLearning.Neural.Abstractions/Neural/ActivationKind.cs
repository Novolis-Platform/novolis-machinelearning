namespace Novolis.MachineLearning.Neural;

/// <summary>Activation functions supported by dense layers.</summary>
public enum ActivationKind
{
    /// <summary>Hyperbolic tangent.</summary>
    Tanh,

    /// <summary>Rectified linear unit.</summary>
    Relu,

    /// <summary>Logistic sigmoid.</summary>
    Sigmoid,

    /// <summary>Identity (no nonlinearity).</summary>
    Linear
}
