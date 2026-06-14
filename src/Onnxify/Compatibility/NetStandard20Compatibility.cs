#if NETSTANDARD2_0
using System.Collections.Generic;

namespace System.Runtime.CompilerServices
{
    internal static class IsExternalInit { }

    [AttributeUsage(AttributeTargets.All, Inherited = false)]
    internal sealed class CompilerFeatureRequiredAttribute : Attribute
    {
        public CompilerFeatureRequiredAttribute(string featureName)
        {
            FeatureName = featureName;
        }

        public string FeatureName { get; }
        public bool IsOptional { get; init; }
    }

    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    internal sealed class RequiredMemberAttribute : Attribute { }

    [AttributeUsage(AttributeTargets.Constructor, AllowMultiple = false, Inherited = false)]
    internal sealed class SetsRequiredMembersAttribute : Attribute { }
}

namespace System
{
    public readonly struct Half : IEquatable<Half>
    {
        public ushort Bits { get; }

        public Half(ushort bits)
        {
            Bits = bits;
        }

        public bool Equals(Half other) => Bits == other.Bits;

        public override bool Equals(object? obj) => obj is Half other && Equals(other);

        public override int GetHashCode() => Bits.GetHashCode();

        public override string ToString() => $"0x{Bits:x4}";
    }
}

namespace System.Linq
{
    internal static class EnumerableCompatibilityExtensions
    {
        public static IEnumerable<TSource> DistinctBy<TSource, TKey>(
            this IEnumerable<TSource> source,
            Func<TSource, TKey> keySelector
        )
        {
            var seen = new HashSet<TKey>();
            foreach (var item in source)
            {
                if (seen.Add(keySelector(item)))
                {
                    yield return item;
                }
            }
        }

        public static HashSet<TSource> ToHashSet<TSource>(
            this IEnumerable<TSource> source,
            IEqualityComparer<TSource>? comparer
        )
        {
            return new HashSet<TSource>(source, comparer);
        }
    }
}

namespace System.Collections.Generic
{
    internal static class DictionaryCompatibilityExtensions
    {
        public static bool TryAdd<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key,
            TValue value
        ) where TKey : notnull
        {
            if (dictionary.ContainsKey(key))
            {
                return false;
            }

            dictionary.Add(key, value);
            return true;
        }

        public static TValue? GetValueOrDefault<TKey, TValue>(
            this IReadOnlyDictionary<TKey, TValue> dictionary,
            TKey key
        )
        {
            return dictionary.TryGetValue(key, out var value) ? value : default;
        }

        public static TValue GetValueOrDefault<TKey, TValue>(
            this IReadOnlyDictionary<TKey, TValue> dictionary,
            TKey key,
            TValue defaultValue
        )
        {
            return dictionary.TryGetValue(key, out var value) ? value : defaultValue;
        }

        public static TValue? GetValueOrDefault<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key
        ) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) ? value : default;
        }

        public static TValue GetValueOrDefault<TKey, TValue>(
            this Dictionary<TKey, TValue> dictionary,
            TKey key,
            TValue defaultValue
        ) where TKey : notnull
        {
            return dictionary.TryGetValue(key, out var value) ? value : defaultValue;
        }
    }
}

namespace Onnxify
{
    internal static class HashCode
    {
        public static int Combine<T1, T2>(T1 value1, T2 value2)
        {
            unchecked
            {
                var hash = 17;
                hash = (hash * 31) + EqualityComparer<T1>.Default.GetHashCode(value1!);
                hash = (hash * 31) + EqualityComparer<T2>.Default.GetHashCode(value2!);
                return hash;
            }
        }
    }

    internal static class ArgumentNullException
    {
        public static void ThrowIfNull(object? argument, string? paramName = null)
        {
            if (argument is null)
            {
                throw new System.ArgumentNullException(paramName);
            }
        }
    }

    internal static class ArgumentOutOfRangeException
    {
        public static void ThrowIfNegative(int value, string? paramName = null)
        {
            if (value < 0)
            {
                throw new System.ArgumentOutOfRangeException(paramName);
            }
        }
    }
}

namespace Onnxify.Data.Numerics
{
    internal static class MathF
    {
        public static float Abs(float value) => Math.Abs(value);

        public static float Pow(float x, float y) => (float)Math.Pow(x, y);

        public static float Pow(float x, int y) => (float)Math.Pow(x, y);

        public static float Log2(float value) => (float)(Math.Log(value) / Math.Log(2d));

        public static float Round(float value) => (float)Math.Round(value);

        public static float Round(float value, MidpointRounding mode) => (float)Math.Round(value, mode);
    }
}
#endif
