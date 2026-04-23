using Onnxify;

namespace Onnxify.Helpers;

/// <summary>
/// Provides internal type-coercion helpers used by ONNX attribute and tensor conversion.
/// </summary>
public static class TypeHelper
{
    internal static T[] NotNull<T>(params T?[] input)
    {
        return input.Where(x => x is not null).Select(x => x!).ToArray();
    }

    internal static TExpected Require<TExpected>(object value)
    {
        if (value is TExpected result)
        {
            return result;
        }

        throw new InvalidCastException(
            $"Expected value of type {typeof(TExpected).FullName}, but got {value.GetType().FullName}."
        );
    }

    internal static IEnumerable<TItem> RequireMany<TItem>(object value)
    {
        if (value is IEnumerable<TItem> result)
        {
            return result;
        }

        throw new InvalidCastException(
            $"Expected value assignable to {typeof(IEnumerable<TItem>).FullName}, but got {value.GetType().FullName}."
        );
    }

    internal static Type? GetCollectionElementType(Type type)
    {
        if (type == typeof(string))
        {
            return null;
        }

        if (type.IsArray)
        {
            return type.GetElementType();
        }

        if (type.IsGenericType)
        {
            var genericArgs = type.GetGenericArguments();

            if (genericArgs.Length == 1 &&
                typeof(IEnumerable<>).MakeGenericType(genericArgs[0]).IsAssignableFrom(type))
            {
                return genericArgs[0];
            }
        }

        var enumerableInterface = type
            .GetInterfaces()
            .FirstOrDefault(x =>
            {
                return
                    x.IsGenericType &&
                    x.GetGenericTypeDefinition() == typeof(IEnumerable<>);
            });

        return enumerableInterface?.GetGenericArguments()[0];
    }
}
