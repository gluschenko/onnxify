namespace Onnxify.Legacy
{
    public abstract class Operator
    {
        public abstract string Name { get; }
        public abstract string Domain { get; }
        public abstract int SinceVersion { get; }
    }

    public interface IFormalParameter
    {
        public string Name { get; }
        public Type Type { get; }
        public FormalParameterOption Option { get; }
        public string? ValueName { get; }
    }

    public class FormalParameter<T> : IFormalParameter
    {
        public string Name { get; }
        public Type Type => typeof(T);
        public FormalParameterOption Option { get; }
        public string? ValueName { get; }

        public FormalParameter(string name, string? valueName)
        {
            Name = name;
            ValueName = valueName;
        }
    }

    public interface IOperatorAttribute
    {
        public string Name { get; }
        public Type Type { get; }
        object? UntypedValue { get; }
    }

    public class OperatorAttribute<T> : IOperatorAttribute
    {
        public string Name { get; }
        public Type Type => typeof(T);
        public T Value { get; }
        public object? UntypedValue => Value;

        public OperatorAttribute(string name, T value)
        {
            Name = name;
            Value = value;
        }
    }

    /// <summary>
    /// From schema.h
    /// </summary>
    public enum FormalParameterOption : byte
    {
        /// <summary>
        /// The formal parameter is single and not optional.
        /// Number of supplied actual parameters must be 1.
        /// </summary>
        Single = 0,
        /// <summary>
        /// The formal parameter is single and optional.
        /// Number of supplied actual parameters may be 0 or 1.
        /// </summary>
        Optional = 1,
        /// <summary>
        /// The formal parameter is variadic.
        /// Number of supplied actual parameters must be N or more, where
        /// the minimum value N is indicated separately (default value 1).
        /// </summary>
        Variadic = 2,
    };

    /// <summary>
    /// From schema.h
    /// </summary>
    public enum DifferentiationCategory : byte
    {
        /// <summary>
        /// Whether this formal parameter is differentiable or not cannot
        /// be statically determined. It also covers variadic formal
        /// parameters which contain both of differentiable and
        /// non-differentiable variables.
        /// </summary>
        Unknown = 0,
        /// <summary>
        /// This formal parameter is differentiable. That is, this formal
        /// parameter can be differentiable input of Gradient operator.
        /// </summary>
        Differentiable = 1,
        /// <summary>
        /// This formal parameter is not differentiable. That is, this formal
        /// parameter can not be differentiable input of Gradient operator.
        /// </summary>
        NonDifferentiable = 2
    };
}
