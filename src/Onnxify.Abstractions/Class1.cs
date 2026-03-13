namespace Onnxify.Abstractions
{
    public abstract class Operator
    {
        public abstract string Name { get; }
        public abstract string Domain { get; }
        public abstract string Doc { get; }
        public abstract int SinceVersion { get; }
    }

    public interface IFormalParameter
    {
        public string Name { get; }
        public Type Type { get; }
        public FormalParameterOption Option { get; }
    }

    public class FormalParameter<T> : IFormalParameter
    {
        public string Name { get; }
        public Type Type => typeof(T);
        public FormalParameterOption Option { get; }
        public T Value { get; }

        public FormalParameter(string name, T value)
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
        // The formal parameter is single and not optional.
        // Number of supplied actual parameters must be 1.
        Single = 0,
        // The formal parameter is single and optional.
        // Number of supplied actual parameters may be 0 or 1.
        Optional = 1,
        // The formal parameter is variadic.
        // Number of supplied actual parameters must be N or more, where
        // the minimum value N is indicated separately (default value 1).
        Variadic = 2,
    };

    /// <summary>
    /// From schema.h
    /// </summary>
    public enum DifferentiationCategory : byte
    {
        // Whether this formal parameter is differentiable or not cannot
        // be statically determined. It also covers variadic formal
        // parameters which contain both of differentiable and
        // non-differentiable variables.
        Unknown = 0,
        // This formal parameter is differentiable. That is, this formal
        // parameter can be differentiable input of Gradient operator.
        Differentiable = 1,
        // This formal parameter is not differentiable. That is, this formal
        // parameter can not be differentiable input of Gradient operator.
        NonDifferentiable = 2
    };
}
