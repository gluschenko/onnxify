using System.Collections.ObjectModel;

namespace Onnxify.Compiler;

internal static class CompilerStructural
{
    public static void RequireNotNull<T>(T value, string parameterName)
    {
        if (value is null)
        {
            throw new ArgumentNullException(parameterName);
        }
    }

    public static IReadOnlyList<T> Copy<T>(IEnumerable<T> values, string parameterName)
    {
        RequireNotNull(values, parameterName);

        var array = values.ToArray();
        for (var index = 0; index < array.Length; index++)
        {
            if (array[index] is null)
            {
                throw new ArgumentException("The collection cannot contain null items.", parameterName);
            }
        }

        var result = new ReadOnlyCollection<T>(array);
        return result;
    }

    public static IReadOnlyList<T>? CopyNullable<T>(IEnumerable<T>? values, string parameterName)
    {
        var result = values is null ? null : Copy(values, parameterName);
        return result;
    }

    public static bool SequenceEqual<T>(IReadOnlyList<T>? left, IReadOnlyList<T>? right)
    {
        if (ReferenceEquals(left, right))
        {
            return true;
        }

        if (left is null || right is null || left.Count != right.Count)
        {
            return false;
        }

        for (var index = 0; index < left.Count; index++)
        {
            if (!EqualityComparer<T>.Default.Equals(left[index], right[index]))
            {
                return false;
            }
        }

        return true;
    }

    public static int GetHashCode<T>(IReadOnlyList<T>? values)
    {
        unchecked
        {
            var hash = 17;
            if (values is null)
            {
                return hash * 31;
            }

            hash = (hash * 31) + values.Count;
            foreach (var value in values)
            {
                var itemHash = value is null
                    ? 0
                    : EqualityComparer<T>.Default.GetHashCode(value);
                hash = (hash * 31) + itemHash;
            }

            return hash;
        }
    }

    public static int Combine(int seed, params object?[] values)
    {
        unchecked
        {
            var hash = seed;
            foreach (var value in values)
            {
                hash = (hash * 31) + (value?.GetHashCode() ?? 0);
            }

            return hash;
        }
    }
}
