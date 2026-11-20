using Onnxify;

namespace Onnxify.Compiler.Operators;

internal abstract class BinaryArithmeticOperator<TNode> : BinaryOperator<TNode>
    where TNode : OnnxNode
{
    private static readonly IReadOnlyCollection<CompilerElementType> REAL_CPU_TYPES = new HashSet<CompilerElementType>
    {
        CompilerElementType.Int8,
        CompilerElementType.Int16,
        CompilerElementType.Int32,
        CompilerElementType.Int64,
        CompilerElementType.Float16,
        CompilerElementType.Float32,
        CompilerElementType.Float64,
    };
    private static readonly IReadOnlyCollection<CompilerElementType> REAL_OR_UINT8_CPU_TYPES = new HashSet<CompilerElementType>(REAL_CPU_TYPES)
    {
        CompilerElementType.UInt8,
    };
    private static readonly IReadOnlyCollection<CompilerElementType> FLOAT_CPU_TYPES = new HashSet<CompilerElementType>
    {
        CompilerElementType.Float16,
        CompilerElementType.Float32,
        CompilerElementType.Float64,
    };
    protected static IReadOnlyCollection<CompilerElementType> RealCpuTypes => REAL_CPU_TYPES;
    protected static IReadOnlyCollection<CompilerElementType> RealOrUInt8CpuTypes => REAL_OR_UINT8_CPU_TYPES;
    protected static IReadOnlyCollection<CompilerElementType> FloatCpuTypes => FLOAT_CPU_TYPES;

    protected BinaryArithmeticOperator(string name, string methodName, string token)
        : base(name, methodName, supportsBroadcast: true, binaryToken: token)
    {
        OperatorTokenValue = token;
    }

    protected string? ValidateSameTypedNumericInputs(
        OnnxNode node,
        IReadOnlyDictionary<string, CompilerType> knownTypes,
        string unsupportedTypeReason,
        IReadOnlyCollection<CompilerElementType>? supportedTypes = null)
    {
        return ValidateKnownInputElementTypes(
            node,
            knownTypes,
            supportedTypes ?? REAL_OR_UINT8_CPU_TYPES,
            unsupportedTypeReason);
    }

    private string OperatorTokenValue { get; }
    protected override string OperatorToken => OperatorTokenValue;
}

