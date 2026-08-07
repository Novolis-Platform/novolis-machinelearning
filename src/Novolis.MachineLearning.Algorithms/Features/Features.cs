using System.Runtime.CompilerServices;

namespace Novolis.MachineLearning.Algorithms;

/// <summary>
/// Immutable fixed-length feature vector with unmanaged value-type elements.
/// </summary>
/// <typeparam name="T">
/// Feature value type. Limited to unmanaged <see cref="IEquatable{T}"/> values so
/// algorithms can rely on cheap copies and stable equality.
/// </typeparam>
public readonly struct Features<T> : IFeatures<T>, IEquatable<Features<T>>
    where T : unmanaged, IEquatable<T>
{
    private readonly T[] _values;

    /// <summary>Creates a feature vector by copying <paramref name="values"/>.</summary>
    /// <param name="values">Non-empty feature values.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> is empty.</exception>
    public Features(ReadOnlySpan<T> values)
    {
        if (values.IsEmpty)
            throw new ArgumentException("At least one feature value is required.", nameof(values));

        _values = values.ToArray();
    }

    /// <summary>Creates a feature vector from the given values.</summary>
    /// <param name="values">Non-empty feature values.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> is empty.</exception>
    public Features(params T[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length == 0)
            throw new ArgumentException("At least one feature value is required.", nameof(values));

        _values = (T[])values.Clone();
    }

    /// <inheritdoc />
    public int Length => _values.Length;

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_values.Length)
                throw new ArgumentOutOfRangeException(nameof(index));

            return _values[index];
        }
    }

    /// <inheritdoc />
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<T> AsSpan() => _values;

    /// <summary>Creates a feature vector from a span.</summary>
    /// <param name="values">Non-empty feature values.</param>
    /// <returns>A new feature vector.</returns>
    public static Features<T> From(ReadOnlySpan<T> values) => new(values);

    /// <inheritdoc />
    public bool Equals(Features<T> other)
    {
        if (_values.Length != other._values.Length)
            return false;

        for (var i = 0; i < _values.Length; i++)
        {
            if (!_values[i].Equals(other._values[i]))
                return false;
        }

        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is Features<T> other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(_values.Length);
        foreach (var value in _values)
            hash.Add(value);
        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public override string ToString() => $"Features<{typeof(T).Name}>[{_values.Length}]";

    /// <summary>Value equality.</summary>
    public static bool operator ==(Features<T> left, Features<T> right) => left.Equals(right);

    /// <summary>Value inequality.</summary>
    public static bool operator !=(Features<T> left, Features<T> right) => !left.Equals(right);
}
