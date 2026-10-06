using System.Globalization;
using System.Reflection;
using System.Text;
using ICSharpCode.Decompiler;
using ICSharpCode.Decompiler.CSharp;
using ICSharpCode.Decompiler.Metadata;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using RoslynLanguageVersion = Microsoft.CodeAnalysis.CSharp.LanguageVersion;

namespace Onnxify.Compiler;

/// <summary>Сеанс импорта и генерации C# TorchSharp через общее промежуточное представление compiler.</summary>
public sealed class CSharpCompilerSession : ICompilerSession
{
    public CompilerResult<ICompilerTree> CreateTree(ICompilerSource source)
    {
        CompilerStructural.RequireNotNull(source, nameof(source));
        if (source is not CSharpTorchSharpSource csharpSource)
        {
            return CompilerResult<ICompilerTree>.Failure(
            [
                new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"The C# compiler session cannot import source kind '{source.Kind}'.",
                    stage: CompilerDiagnosticStage.Parse,
                    severity: CompilerDiagnosticSeverity.Error),
            ]);
        }

        var importResult = CSharpCompilerFrontend.Import(csharpSource);
        var treeResult = CompilerResultMapper.Map(
            source: importResult,
            projection: static computationTree => (ICompilerTree)computationTree,
            missingValueStage: CompilerDiagnosticStage.Analyze,
            missingValueMessage: "The C# frontend reported success without producing a computation tree.");
        return treeResult;
    }

    public CompilerResult<TOutput> Generate<TOutput>(
        ICompilerTree tree,
        ICompilerSink<TOutput> sink
    )
    {
        CompilerStructural.RequireNotNull(tree, nameof(tree));
        CompilerStructural.RequireNotNull(sink, nameof(sink));

        if (sink is CSharpCompilerSink && typeof(TOutput) == typeof(string))
        {
            if (tree is not CompilerComputationTree computationTree)
            {
                return CompilerResult<TOutput>.Failure(
                [
                    new CompilerDiagnostic(
                        code: CompilerDiagnosticCodes.InvalidSource,
                        message: "The C# backend requires a CompilerComputationTree.",
                        stage: CompilerDiagnosticStage.Emit,
                        severity: CompilerDiagnosticSeverity.Error),
                ]);
            }

            var generated = CSharpCompilerBackend.Generate(computationTree);
            var resultSource = CompilerResultMapper.Map(
                source: generated,
                projection: static generatedSource => (TOutput)(object)generatedSource,
                missingValueStage: CompilerDiagnosticStage.Emit,
                missingValueMessage: "The C# backend reported success without producing source text.");
            return resultSource;
        }

        return CompilerResult<TOutput>.Failure(
        [
            new CompilerDiagnostic(
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"The C# compiler session cannot emit target '{sink.Kind}'.",
                stage: CompilerDiagnosticStage.Emit,
                severity: CompilerDiagnosticSeverity.Error),
        ]);
    }
}

internal static class CSharpCompilerFrontend
{
    /// <summary>
    /// Разбирает заданный исходник или декомпилирует указанный метод и преобразует его в промежуточное представление.
    /// Ошибки анализа возвращаются как diagnostics, чтобы вызывающая сторона не зависела от Roslyn и ILSpy exceptions.
    /// </summary>
    public static CompilerResult<CompilerComputationTree> Import(CSharpTorchSharpSource source)
    {
        try
        {
            if (source.Module is not null)
            {
                return ImportCompiledModule(source.Module);
            }

            return ImportSourceText(source.SourceText, "<memory>", null);
        }
        catch (CSharpCompilerDiagnosticException exception)
        {
            return CompilerResult<CompilerComputationTree>.Failure(
            [
                new CompilerDiagnostic(
                    code: exception.Code,
                    message: exception.Message,
                    stage: exception.Stage,
                    severity: CompilerDiagnosticSeverity.Error,
                    span: exception.Span,
                    context: exception.Context),
            ]);
        }
        catch (Exception exception)
        {
            return CompilerResult<CompilerComputationTree>.Failure(
            [
                new CompilerDiagnostic(
                    code: CompilerDiagnosticCodes.InvalidSource,
                    message: $"The C# source could not be imported: {exception.Message}",
                    stage: CompilerDiagnosticStage.Parse,
                    severity: CompilerDiagnosticSeverity.Error),
            ]);
        }
    }

    /// <summary>
    /// Декомпилирует выбранный метод и объявленные локальные helpers в общий C# source tree для scanner.
    /// Отдельная декомпиляция каждого метода сохраняет compiler-owned boundary и доступность helper bodies.
    /// </summary>
    private static CompilerResult<CompilerComputationTree> ImportCompiledModule(
        CompilerTorchSharpModuleDescriptor descriptor
    )
    {
        if (!File.Exists(descriptor.AssemblyPath))
        {
            throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: $"Assembly '{descriptor.AssemblyPath}' does not exist.",
                stage: CompilerDiagnosticStage.Parse,
                span: Span(descriptor.Document, 0, 0));
        }

        var decompiledMethods = new List<MethodDeclarationSyntax>();
        AppendModuleMethods(
            descriptor: descriptor,
            prefix: string.Empty,
            decompiledMethods: decompiledMethods);
        var sourceText = $"public sealed class DecompiledTorchSharpModule {{ {string.Join(Environment.NewLine, decompiledMethods)} }}";
        return ImportSourceText(
            sourceText,
            descriptor.Document,
            new CompilationContext(descriptor));
    }

    private static void AppendModuleMethods(
        CompilerTorchSharpModuleDescriptor descriptor,
        string prefix,
        List<MethodDeclarationSyntax> decompiledMethods
    )
    {
        var assembly = Assembly.LoadFrom(descriptor.AssemblyPath);
        var moduleType = assembly.GetType(descriptor.TypeName, throwOnError: false, ignoreCase: false);
        if (moduleType is null)
        {
            throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: $"Type '{descriptor.TypeName}' was not found in '{descriptor.AssemblyPath}'.",
                stage: CompilerDiagnosticStage.Parse,
                span: Span(descriptor.Document, 0, 0));
        }

        var settings = new DecompilerSettings(ICSharpCode.Decompiler.CSharp.LanguageVersion.CSharp10_0)
        {
            ThrowOnAssemblyResolveErrors = false,
        };
        var decompiler = new CSharpDecompiler(descriptor.AssemblyPath, settings);
        var rootMethod = FindMethod(moduleType, descriptor);
        AddMethod(rootMethod, GetBlockName(prefix, descriptor.MethodName));
        foreach (var helperDescriptor in descriptor.HelperMethods)
        {
            var helperMethod = moduleType
                .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                .Where(candidate => candidate.DeclaringType == moduleType && !candidate.IsSpecialName)
                .Where(candidate => string.Equals(candidate.Name, helperDescriptor.Name, StringComparison.Ordinal))
                .OrderBy(candidate => candidate.MetadataToken)
                .FirstOrDefault();
            if (helperMethod is not null)
            {
                AddMethod(helperMethod, GetBlockName(prefix, helperDescriptor.Name));
            }
        }

        foreach (var childModule in descriptor.ChildModules)
        {
            if (childModule.Module is null)
            {
                continue;
            }

            AppendModuleMethods(
                descriptor: childModule.Module,
                prefix: GetBlockName(prefix, childModule.BlockName) + "__",
                decompiledMethods: decompiledMethods);
        }

        void AddMethod(MethodInfo method, string generatedName)
        {
            var syntaxTree = decompiler.Decompile(MetadataTokenHelpers.EntityHandleOrNil(method.MetadataToken));
            var declaration = syntaxTree
                .Descendants
                .OfType<ICSharpCode.Decompiler.CSharp.Syntax.MethodDeclaration>()
                .FirstOrDefault(candidate => string.Equals(candidate.Name, method.Name, StringComparison.Ordinal));
            if (declaration is null)
            {
                if (ReferenceEquals(method, rootMethod))
                {
                    throw new CSharpCompilerDiagnosticException(
                        code: CompilerDiagnosticCodes.InvalidSource,
                        message: $"Method '{descriptor.MethodName}' could not be decompiled.",
                        stage: CompilerDiagnosticStage.Parse,
                        span: Span(descriptor.Document, 0, 0));
                }

                return;
            }

            var methodText = declaration.ToString();
            var roslynMethod = CSharpSyntaxTree.ParseText(
                    $"class DecompiledMethod {{ {methodText} }}",
                    new CSharpParseOptions(RoslynLanguageVersion.Latest))
                .GetRoot()
                .DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .First();
            var rewriter = new CompiledModuleMethodRewriter(
                generatedName: generatedName,
                helperNames: descriptor.HelperMethods.Select(helper => helper.Name),
                childModules: descriptor.ChildModules,
                prefix: prefix);
            if (rewriter.Visit(roslynMethod) is MethodDeclarationSyntax rewrittenMethod)
            {
                decompiledMethods.Add(rewrittenMethod);
            }
            else
            {
                throw new CSharpCompilerDiagnosticException(
                    code: CompilerDiagnosticCodes.InvalidSource,
                    message: $"Method '{method.Name}' could not be normalized for compiler scanning.",
                    stage: CompilerDiagnosticStage.Parse,
                    span: Span(descriptor.Document, 0, 0));
            }
        }
    }

    internal static string GetBlockName(string prefix, string name)
    {
        return string.IsNullOrEmpty(prefix) ? name : $"{prefix}{name}";
    }

    private static MethodInfo FindMethod(
        Type moduleType,
        CompilerTorchSharpModuleDescriptor descriptor
    )
    {
        if (descriptor.MethodMetadataToken is int token)
        {
            var tokenMethod = moduleType
                .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                .FirstOrDefault(x => x.MetadataToken == token);
            if (tokenMethod is not null)
            {
                return tokenMethod;
            }
        }

        return moduleType
            .GetMethods(BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
            .Where(x => string.Equals(x.Name, descriptor.MethodName, StringComparison.Ordinal))
            .OrderBy(x => x.MetadataToken)
            .FirstOrDefault()
            ?? throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: $"Method '{descriptor.MethodName}' was not found on '{moduleType.FullName}'.",
                stage: CompilerDiagnosticStage.Parse,
                span: Span(descriptor.Document, 0, 0));
    }

    private static CompilerResult<CompilerComputationTree> ImportSourceText(
        string sourceText,
        string document,
        CompilationContext? context
    )
    {
        var tree = CSharpSyntaxTree.ParseText(
            sourceText,
            new CSharpParseOptions(RoslynLanguageVersion.Latest),
            document);
        var root = tree.GetRoot();
        var method = root.DescendantNodes().OfType<MethodDeclarationSyntax>().FirstOrDefault();
        var initialParseError = tree.GetDiagnostics().FirstOrDefault(x => x.Severity == DiagnosticSeverity.Error);

        if (method is null || initialParseError is not null)
        {
            sourceText = method is null && !LooksLikeMethod(sourceText)
                ? WrapBody(sourceText)
                : WrapMethod(sourceText);
            tree = CSharpSyntaxTree.ParseText(
                sourceText,
                new CSharpParseOptions(RoslynLanguageVersion.Latest),
                document);
            root = tree.GetRoot();
            method = root.DescendantNodes().OfType<MethodDeclarationSyntax>().First();
        }

        var parseError = tree.GetDiagnostics().FirstOrDefault(x => x.Severity == DiagnosticSeverity.Error);
        if (parseError is not null)
        {
            throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: parseError.GetMessage(),
                stage: CompilerDiagnosticStage.Parse,
                span: Span(document, parseError.Location));
        }

        var scanner = new CSharpSyntaxScanner(document, context);
        var result = scanner.Scan(
            method,
            root.DescendantNodes()
                .OfType<MethodDeclarationSyntax>()
                .Where(x => !ReferenceEquals(x, method)));
        return result;
    }

    private static string WrapBody(string sourceText)
    {
        return "public sealed class CompilerSource { public global::TorchSharp.torch.Tensor forward(global::TorchSharp.torch.Tensor input) {\n"
            + sourceText
            + "\n} }";
    }

    private static string WrapMethod(string sourceText)
    {
        return "public sealed class CompilerSource {\n"
            + sourceText
            + "\n}";
    }

    private static bool LooksLikeMethod(string sourceText)
    {
        return sourceText.Contains("forward(", StringComparison.Ordinal)
            || sourceText.Contains(" forward (", StringComparison.Ordinal)
            || sourceText.Contains(" forward(", StringComparison.Ordinal);
    }

    internal static CompilerSourceSpan Span(
        string document,
        int start,
        int length
    )
    {
        return new CompilerSourceSpan(
            kind: CompilerSourceSpanKind.CSharp,
            document: document,
            start: start,
            length: length,
            startLine: 1,
            startColumn: 1,
            endLine: 1,
            endColumn: Math.Max(1, length + 1));
    }

    internal static CompilerSourceSpan Span(
        string document,
        Location location
    )
    {
        if (location == Location.None)
        {
            return Span(document, 0, 0);
        }

        var lineSpan = location.GetLineSpan();
        return new CompilerSourceSpan(
            kind: CompilerSourceSpanKind.CSharp,
            document: string.IsNullOrWhiteSpace(lineSpan.Path) ? document : lineSpan.Path,
            start: location.SourceSpan.Start,
            length: location.SourceSpan.Length,
            startLine: lineSpan.StartLinePosition.Line + 1,
            startColumn: lineSpan.StartLinePosition.Character + 1,
            endLine: lineSpan.EndLinePosition.Line + 1,
            endColumn: lineSpan.EndLinePosition.Character + 1);
    }

    internal sealed class CompilationContext
    {
        public CompilationContext(
            CompilerTorchSharpModuleDescriptor descriptor)
        {
            Descriptor = descriptor;
        }

        public CompilerTorchSharpModuleDescriptor Descriptor { get; }
    }
}

internal sealed class CompiledModuleMethodRewriter : CSharpSyntaxRewriter
{
    private readonly string _generatedName;
    private readonly HashSet<string> _helperNames;
    private readonly IReadOnlyDictionary<string, string> _childModuleNames;
    private readonly string _prefix;

    public CompiledModuleMethodRewriter(
        string generatedName,
        IEnumerable<string> helperNames,
        IEnumerable<CompilerTorchSharpChildModuleDescriptor> childModules,
        string prefix
    )
    {
        _generatedName = generatedName;
        _prefix = prefix;
        _helperNames = new HashSet<string>(helperNames, StringComparer.Ordinal);
        _childModuleNames = childModules.ToDictionary(
            child => child.Name,
            child => CSharpCompilerFrontend.GetBlockName(_prefix + child.BlockName + "__", "forward"),
            StringComparer.Ordinal);
    }

    public override SyntaxNode? VisitMethodDeclaration(MethodDeclarationSyntax node)
    {
        return base.VisitMethodDeclaration(node.WithIdentifier(SyntaxFactory.Identifier(_generatedName)));
    }

    public override SyntaxNode? VisitInvocationExpression(InvocationExpressionSyntax node)
    {
        var expression = node.Expression;
        if (expression is IdentifierNameSyntax identifier
            && _helperNames.Contains(identifier.Identifier.ValueText))
        {
            expression = SyntaxFactory.IdentifierName(CSharpCompilerFrontend.GetBlockName(_prefix, identifier.Identifier.ValueText));
        }
        else if (expression is MemberAccessExpressionSyntax member)
        {
            var memberName = member.Name.Identifier.ValueText;
            if (_helperNames.Contains(memberName)
                && TryGetMemberPath(member.Expression, out var helperTarget)
                && helperTarget == "this")
            {
                expression = SyntaxFactory.IdentifierName(CSharpCompilerFrontend.GetBlockName(_prefix, memberName));
            }
            else if (string.Equals(memberName, "forward", StringComparison.Ordinal)
                && TryGetMemberPath(member.Expression, out var childPath)
                && _childModuleNames.TryGetValue(NormalizeChildPath(childPath), out var childMethodName))
            {
                expression = SyntaxFactory.IdentifierName(childMethodName);
            }
        }

        return base.VisitInvocationExpression(node.WithExpression(expression));
    }

    private static bool TryGetMemberPath(ExpressionSyntax expression, out string path)
    {
        switch (expression)
        {
            case ThisExpressionSyntax:
                path = "this";
                return true;
            case IdentifierNameSyntax identifier:
                path = identifier.Identifier.ValueText;
                return true;
            case MemberAccessExpressionSyntax member when TryGetMemberPath(member.Expression, out var prefix):
                path = $"{prefix}.{member.Name.Identifier.ValueText}";
                return true;
            default:
                path = string.Empty;
                return false;
        }
    }

    private static string NormalizeChildPath(string path)
    {
        const string THIS_PREFIX = "this.";
        return path.StartsWith(THIS_PREFIX, StringComparison.Ordinal)
            ? path.Substring(THIS_PREFIX.Length)
            : path;
    }
}

internal sealed class CSharpSyntaxScanner
{
    private static readonly CompilerType DEFAULT_TENSOR_TYPE = new CompilerTensorType(
        CompilerElementType.Float32,
        dimensions: null);

    private readonly string _document;
    private readonly CSharpCompilerFrontend.CompilationContext? _context;

    public CSharpSyntaxScanner(
        string document,
        CSharpCompilerFrontend.CompilationContext? context
    )
    {
        _document = document;
        _context = context;
    }

    /// <summary>
    /// Сканирует выбранный method body и переносит контракты, state и helper методы в промежуточное представление compiler.
    /// Decompiler и Roslyn остаются внутри frontend boundary; наружу выходит только immutable tree.
    /// </summary>
    public CompilerResult<CompilerComputationTree> Scan(
        MethodDeclarationSyntax method,
        IEnumerable<MethodDeclarationSyntax>? helperMethods = null
    )
    {
        var descriptor = _context?.Descriptor;
        var inputs = GetInputs(method, descriptor);
        var outputs = GetOutputs(method, descriptor);
        ValidateNoRecursiveCall(method);
        ValidateNoDynamicModuleDispatch(method);
        var body = ScanBody(method);
        (inputs, outputs) = ApplyOperatorTypeContracts(body, inputs, outputs);
        var builder = CreateBuilder(method, descriptor);
        RegisterValueContracts(builder, inputs, outputs);
        RegisterStateMetadata(builder, descriptor);

        AddDeclaredValues(
            builder: builder,
            body: body,
            inputs: inputs,
            outputs: outputs,
            descriptor: descriptor);
        var helperBlocks = AddHelperBlocks(builder, helperMethods);

        if (!TryLowerMappedOperation(method, body, inputs, outputs, helperBlocks, builder))
        {
            builder.SetSyntaxBody(body);
        }

        try
        {
            var tree = builder.Build();
            var result = CompilerResult<CompilerComputationTree>.Success(tree);
            return result;
        }
        catch (ArgumentException exception)
        {
            throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.DuplicateName,
                message: exception.Message,
                stage: CompilerDiagnosticStage.Analyze,
                span: Span(method));
        }
    }

    private bool TryLowerMappedOperation(
        MethodDeclarationSyntax method,
        CompilerBlockStatement body,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs,
        IReadOnlyDictionary<string, CompilerComputationBlock> helperBlocks,
        CompilerComputationTreeBuilder builder
    )
    {
        if (TryGetDeconstructionHelperCall(body, outputs.Count, out var deconstructionCall)
            && deconstructionCall is not null
            && TryGetReferenceName(deconstructionCall.Target, out var deconstructionHelperName)
            && helperBlocks.TryGetValue(deconstructionHelperName, out var deconstructionHelper))
        {
            AddHelperCall(
                method: method,
                helper: deconstructionHelper,
                helperName: deconstructionHelperName,
                invocation: deconstructionCall,
                inputs: inputs,
                outputs: outputs,
                body: body,
                builder: builder);
            return true;
        }

        if (!TryGetStaticReturnExpression(body, out var returnExpression))
        {
            return false;
        }

        if (returnExpression is CompilerBinaryExpression binaryExpression
            && CompilerOperatorMappingRegistry.TryGetTorchSharpBinaryOperator(
                binaryExpression.Operator,
                out var binaryMapping)
            && binaryMapping is not null)
        {
            AddBinaryOperation(
                method,
                binaryExpression,
                binaryMapping,
                inputs,
                outputs,
                builder);
            return true;
        }

        if (returnExpression is CompilerUnaryExpression unaryExpression
            && CompilerOperatorMappingRegistry.TryGetTorchSharpUnaryOperator(
                unaryExpression.Operator,
                out var unaryMapping)
            && unaryMapping is not null)
        {
            AddUnaryOperation(method, unaryExpression, unaryMapping, inputs, outputs, builder);
            return true;
        }

        if (returnExpression is not CompilerInvocationExpression invocation)
        {
            return false;
        }

        if (TryGetReferenceName(invocation.Target, out var helperName)
            && helperBlocks.TryGetValue(helperName, out var helper))
        {
            AddHelperCall(method, helper, helperName, invocation, inputs, outputs, body, builder);
            return true;
        }

        if (!CompilerOperatorMappingRegistry.TryGetTorchSharpCall(invocation.Target, out var mapping, out var receiver)
            || mapping is null)
        {
            return false;
        }

        if (mapping.OnnxName is "MatMul" or "Gemm")
        {
            AddMatrixOperation(method, invocation, mapping, receiver, inputs, outputs, builder);
            return true;
        }

        var operationInputs = new List<CompilerExpression>();
        if (receiver is not null)
        {
            operationInputs.Add(receiver);
        }

        var requiredInvocationInputs = mapping.InputCount - operationInputs.Count;
        if (requiredInvocationInputs < 0
            || invocation.Arguments.Count < requiredInvocationInputs
            || invocation.Arguments.Count < requiredInvocationInputs + mapping.FixedTorchSharpArguments.Count
            || invocation.Arguments.Count > requiredInvocationInputs + mapping.AttributeNames.Count + mapping.FixedTorchSharpArguments.Count
            || outputs.Count != 1)
        {
            throw Unsupported(
                method,
                $"TorchSharp mapping '{mapping.TorchSharpNames[0]}' requires {(mapping.InputCount == 1 ? "one tensor input" : $"exactly {mapping.InputCount} tensor inputs")}, one output, and supported literal activation arguments.");
        }

        operationInputs.AddRange(invocation.Arguments.Take(requiredInvocationInputs));
        var operationInputReferences = new List<CompilerValueReference>();
        for (var index = 0; index < operationInputs.Count; index++)
        {
            var expression = operationInputs[index];
            if (expression is CompilerReferenceExpression reference
                && inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)))
            {
                operationInputReferences.Add(new CompilerValueReference(reference.Name));
                continue;
            }

            if (TryAddScalarInitializer(
                method,
                mapping,
                index,
                expression,
                inputs,
                builder,
                out var initializerReference))
            {
                operationInputReferences.Add(initializerReference);
                continue;
            }

            throw Unsupported(method, $"TorchSharp mapping '{mapping.TorchSharpNames[0]}' requires tensor inputs or scalar literal operands.");
        }

        var attributes = new List<CompilerAttribute>();
        foreach (var (argument, index) in invocation.Arguments
            .Skip(requiredInvocationInputs)
            .Take(mapping.AttributeNames.Count)
            .Select((argument, index) => (argument, index)))
        {
            if (mapping.OnnxName == "Cast"
                && TryGetMemberPath(argument, out var castTypeName)
                && TryGetCastType(castTypeName, out var onnxCastType))
            {
                attributes.Add(new CompilerAttribute(
                    mapping.AttributeNames[index],
                    new CompilerSignedIntegerLiteral(CompilerElementType.Int64, onnxCastType)));
                continue;
            }

            if (argument is not CompilerLiteralExpression literalExpression)
            {
                throw Unsupported(method, $"TorchSharp mapping '{mapping.TorchSharpNames[0]}' requires compile-time literal activation arguments.");
            }

            attributes.Add(new CompilerAttribute(mapping.AttributeNames[index], literalExpression.Literal));
        }

        foreach (var (argument, index) in invocation.Arguments
            .Skip(invocation.Arguments.Count - mapping.FixedTorchSharpArguments.Count)
            .Select((argument, index) => (argument, index)))
        {
            if (argument is not CompilerLiteralExpression { Literal: CompilerFloatingPointLiteral fixedValue }
                || (float)fixedValue.Value != mapping.FixedTorchSharpArguments[index])
            {
                throw Unsupported(method, $"TorchSharp mapping '{mapping.TorchSharpNames[0]}' only supports its fixed trailing arguments.");
            }
        }

        builder.AddOperation(new CompilerOperation(
            name: mapping.OnnxName.ToLowerInvariant(),
            descriptor: mapping.Descriptor,
            inputs: operationInputReferences,
            outputs: [new CompilerValueReference(outputs[0].Name)],
            attributes: attributes,
            span: invocation.Span));
        return true;
    }

    private void AddMatrixOperation(
        MethodDeclarationSyntax method,
        CompilerInvocationExpression invocation,
        CompilerOperatorMapping mapping,
        CompilerExpression? receiver,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs,
        CompilerComputationTreeBuilder builder
    )
    {
        if (outputs.Count != 1)
        {
            throw Unsupported(method, $"TorchSharp mapping '{mapping.OnnxName}' requires exactly one output.");
        }

        var callName = CompilerOperatorMappingRegistry.GetTorchSharpCallName(invocation.Target, receiver);
        var operationInputs = new List<CompilerValueReference>();
        var attributes = new List<CompilerAttribute>();
        if (mapping.OnnxName == "MatMul")
        {
            var operands = new List<CompilerExpression>();
            if (receiver is not null)
            {
                operands.Add(receiver);
            }

            operands.AddRange(invocation.Arguments);
            if (operands.Count != 2)
            {
                throw Unsupported(method, $"TorchSharp call '{callName}' requires exactly two tensor operands.");
            }

            operationInputs.AddRange(operands.Select(operand => RequireMatrixTensorReference(method, operand, inputs, callName)));
        }
        else if (callName == "torch.nn.functional.linear")
        {
            if (receiver is not null || invocation.Arguments.Count is < 2 or > 3)
            {
                throw Unsupported(method, "torch.nn.functional.linear requires input, weight, and an optional bias tensor.");
            }

            operationInputs.Add(RequireMatrixTensorReference(method, invocation.Arguments[0], inputs, callName));
            operationInputs.Add(RequireMatrixTensorReference(method, invocation.Arguments[1], inputs, callName));
            if (invocation.Arguments.Count == 3)
            {
                operationInputs.Add(RequireMatrixTensorReference(method, invocation.Arguments[2], inputs, callName));
            }

            attributes.Add(IntegerAttribute("transB", 1));
        }
        else
        {
            var arguments = new List<CompilerExpression>();
            if (receiver is not null)
            {
                arguments.Add(receiver);
            }

            arguments.AddRange(invocation.Arguments);
            if (arguments.Count is < 3 or > 5)
            {
                throw Unsupported(method, "torch.addmm requires input, mat1, mat2, and optional constant beta and alpha values.");
            }

            var c = RequireMatrixTensorReference(method, arguments[0], inputs, callName);
            var a = RequireMatrixTensorReference(method, arguments[1], inputs, callName);
            var b = RequireMatrixTensorReference(method, arguments[2], inputs, callName);
            operationInputs.Add(a);
            operationInputs.Add(b);
            operationInputs.Add(c);
            var beta = arguments.Count >= 4 ? RequireNumericLiteral(method, arguments[3], "beta") : 1f;
            var alpha = arguments.Count >= 5 ? RequireNumericLiteral(method, arguments[4], "alpha") : 1f;
            attributes.Add(FloatAttribute("alpha", alpha));
            attributes.Add(FloatAttribute("beta", beta));
        }

        builder.AddOperation(new CompilerOperation(
            name: mapping.OnnxName.ToLowerInvariant(),
            descriptor: mapping.Descriptor,
            inputs: operationInputs,
            outputs: [new CompilerValueReference(outputs[0].Name)],
            attributes: attributes,
            span: invocation.Span));
    }

    private CompilerValueReference RequireMatrixTensorReference(
        MethodDeclarationSyntax method,
        CompilerExpression expression,
        IReadOnlyList<ScanValue> inputs,
        string callName
    )
    {
        if (expression is CompilerReferenceExpression reference
            && inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)))
        {
            return new CompilerValueReference(reference.Name);
        }

        throw Unsupported(method, $"TorchSharp call '{callName}' requires tensor parameters for its matrix operands.");
    }

    private float RequireNumericLiteral(MethodDeclarationSyntax method, CompilerExpression expression, string parameterName)
    {
        if (expression is CompilerLiteralExpression literal)
        {
            return literal.Literal switch
            {
                CompilerFloatingPointLiteral floatingPoint => (float)floatingPoint.Value,
                CompilerSignedIntegerLiteral integer => integer.Value,
                CompilerUnsignedIntegerLiteral integer => integer.Value,
                _ => throw Unsupported(method, $"Gemm parameter '{parameterName}' must be a numeric compile-time constant."),
            };
        }

        throw Unsupported(method, $"Gemm parameter '{parameterName}' must be a numeric compile-time constant.");
    }

    private static CompilerAttribute FloatAttribute(string name, float value) => new(
        name,
        new CompilerFloatingPointLiteral(CompilerElementType.Float32, value));

    private static CompilerAttribute IntegerAttribute(string name, long value) => new(
        name,
        new CompilerSignedIntegerLiteral(CompilerElementType.Int64, value));

    private void AddBinaryOperation(
        MethodDeclarationSyntax method,
        CompilerBinaryExpression expression,
        CompilerOperatorMapping mapping,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs,
        CompilerComputationTreeBuilder builder
    )
    {
        if (mapping.InputCount != 2 || outputs.Count != 1)
        {
            throw Unsupported(method, $"TorchSharp operator mapping '{mapping.OnnxName}' requires two inputs and one output.");
        }

        var inputExpressions = new[] { expression.Left, expression.Right };
        var references = new List<CompilerValueReference>();
        foreach (var (inputExpression, index) in inputExpressions.Select((value, index) => (value, index)))
        {
            if (inputExpression is CompilerReferenceExpression reference
                && inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)))
            {
                references.Add(new CompilerValueReference(reference.Name));
                continue;
            }

            if (TryAddScalarInitializer(method, mapping, index, inputExpression, inputs, builder, out var initializerReference))
            {
                references.Add(initializerReference);
                continue;
            }

            throw Unsupported(method, $"TorchSharp operator mapping '{mapping.OnnxName}' requires tensor inputs or a scalar literal paired with a tensor input.");
        }

        if (references.Count != 2)
        {
            throw Unsupported(method, $"TorchSharp operator mapping '{mapping.OnnxName}' requires at least one tensor input.");
        }

        builder.AddOperation(new CompilerOperation(
            name: mapping.OnnxName.ToLowerInvariant(),
            descriptor: mapping.Descriptor,
            inputs: references,
            outputs: [new CompilerValueReference(outputs[0].Name)],
            span: expression.Span));
    }

    private void AddUnaryOperation(
        MethodDeclarationSyntax method,
        CompilerUnaryExpression expression,
        CompilerOperatorMapping mapping,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs,
        CompilerComputationTreeBuilder builder
    )
    {
        if (mapping.InputCount != 1
            || outputs.Count != 1
            || expression.Expression is not CompilerReferenceExpression reference
            || !inputs.Any(input => string.Equals(input.Name, reference.Name, StringComparison.Ordinal)))
        {
            throw Unsupported(method, $"TorchSharp operator mapping '{mapping.OnnxName}' requires one tensor input and one output.");
        }

        builder.AddOperation(new CompilerOperation(
            name: mapping.OnnxName.ToLowerInvariant(),
            descriptor: mapping.Descriptor,
            inputs: [new CompilerValueReference(reference.Name)],
            outputs: [new CompilerValueReference(outputs[0].Name)],
            span: expression.Span));
    }

    private bool TryAddScalarInitializer(
        MethodDeclarationSyntax method,
        CompilerOperatorMapping mapping,
        int index,
        CompilerExpression expression,
        IReadOnlyList<ScanValue> inputs,
        CompilerComputationTreeBuilder builder,
        out CompilerValueReference reference
    )
    {
        if (expression is CompilerLiteralExpression { Literal: CompilerScalarLiteral scalarLiteral })
        {
            var elementType = inputs
                .Select(input => input.Type)
                .OfType<CompilerTensorType>()
                .Select(tensorType => (CompilerElementType?)tensorType.ElementType)
                .FirstOrDefault();
            if (elementType is null)
            {
                throw Unsupported(method, $"TorchSharp mapping '{mapping.OnnxName}' cannot infer scalar dtype without a tensor parameter.");
            }

            var name = $"__{mapping.OnnxName.ToLowerInvariant()}_scalar{index}";
            var value = ConvertScalarLiteral(scalarLiteral, elementType.Value, method);
            var tensorLiteral = new CompilerTensorLiteral(
                elementType.Value,
                Array.Empty<CompilerDimension>(),
                [value]);
            builder.AddStateMember(new CompilerStateMember(
                name,
                CompilerStateMemberKind.Initializer,
                new CompilerTensorType(elementType.Value, Array.Empty<CompilerDimension>()),
                tensorLiteral));
            reference = new CompilerValueReference(name);
            return true;
        }

        reference = null!;
        return false;
    }

    private CompilerScalarLiteral ConvertScalarLiteral(
        CompilerScalarLiteral literal,
        CompilerElementType targetType,
        SyntaxNode source
    )
    {
        try
        {
            return targetType switch
            {
                CompilerElementType.Float16 or CompilerElementType.BFloat16 or CompilerElementType.Float32 or CompilerElementType.Float64
                    => new CompilerFloatingPointLiteral(targetType, Convert.ToDouble(ScalarValue(literal), CultureInfo.InvariantCulture)),
                CompilerElementType.Int8 or CompilerElementType.Int16 or CompilerElementType.Int32 or CompilerElementType.Int64
                    => new CompilerSignedIntegerLiteral(targetType, Convert.ToInt64(ScalarValue(literal), CultureInfo.InvariantCulture)),
                CompilerElementType.UInt8 or CompilerElementType.UInt16 or CompilerElementType.UInt32 or CompilerElementType.UInt64
                    => new CompilerUnsignedIntegerLiteral(targetType, Convert.ToUInt64(ScalarValue(literal), CultureInfo.InvariantCulture)),
                _ => throw Unsupported(source, $"Scalar operands are unsupported for tensor element type '{targetType}'."),
            };
        }
        catch (Exception exception) when (exception is FormatException or InvalidCastException or OverflowException)
        {
            throw Unsupported(source, $"Scalar operand cannot be represented as '{targetType}'.");
        }
    }

    private static object ScalarValue(CompilerScalarLiteral literal) => literal switch
    {
        CompilerBooleanLiteral boolean => boolean.Value,
        CompilerSignedIntegerLiteral integer => integer.Value,
        CompilerUnsignedIntegerLiteral integer => integer.Value,
        CompilerFloatingPointLiteral floating => floating.Value,
        _ => throw new InvalidCastException(),
    };

    private void AddHelperCall(
        MethodDeclarationSyntax method,
        CompilerComputationBlock helper,
        string helperName,
        CompilerInvocationExpression invocation,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs,
        CompilerBlockStatement body,
        CompilerComputationTreeBuilder builder
    )
    {
        var helperArguments = new List<CompilerExpression>();
        var availableValues = new HashSet<string>(
            inputs.Select(input => input.Name)
                .Concat(FindDeclarations(body).Select(declaration => declaration.Name)),
            StringComparer.Ordinal);
        foreach (var argument in invocation.Arguments)
        {
            if (argument is CompilerReferenceExpression reference
                && availableValues.Contains(reference.Name))
            {
                helperArguments.Add(argument);
                continue;
            }

            if (argument is CompilerLiteralExpression { Literal: CompilerSignedIntegerLiteral or CompilerUnsignedIntegerLiteral or CompilerFloatingPointLiteral or CompilerBooleanLiteral })
            {
                helperArguments.Add(argument);
                continue;
            }

            if (argument is not CompilerReferenceExpression)
            {
                throw Unsupported(
                    method,
                    $"Helper call '{helperName}' only supports value references and scalar literal arguments.");
            }

            throw Unsupported(
                method,
                $"Helper call '{helperName}' references a value that is not available in the forward scope.");
        }

        if (helperArguments.Count != helper.Inputs.Count || outputs.Count != helper.Outputs.Count)
        {
            throw Unsupported(
                method,
                $"Helper call '{helperName}' has incompatible input or output bindings.");
        }

        builder.AddOperation(CompilerModuleCall.CreateWithArguments(
            name: $"{helperName}__call0",
            targetBlock: helper.Name,
            arguments: helperArguments,
            outputs: outputs.Select(output => new CompilerValueReference(output.Name)),
            span: invocation.Span));
    }

    private static bool TryGetDeconstructionHelperCall(
        CompilerBlockStatement body,
        int outputCount,
        out CompilerInvocationExpression? invocation
    )
    {
        invocation = null;
        if (body.Statements.Count != 2
            || body.Statements[0] is not CompilerBlockStatement declarations
            || declarations.Statements.Count != outputCount
            || body.Statements[1] is not CompilerReturnStatement { Expression: CompilerTupleExpression returnedTuple }
            || returnedTuple.Items.Count != outputCount)
        {
            return false;
        }

        var declarationStatements = declarations.Statements.OfType<CompilerDeclarationStatement>().ToArray();
        if (declarationStatements.Length != outputCount
            || declarationStatements.Any(declaration => declaration.Initializer is not CompilerMemberAccessExpression
            {
                Target: CompilerInvocationExpression,
            }))
        {
            return false;
        }

        var indexers = declarationStatements
            .Select(declaration => declaration.Initializer)
            .OfType<CompilerMemberAccessExpression>()
            .ToArray();
        if (indexers.Select((indexer, index) => string.Equals(indexer.MemberName, $"Item{index + 1}", StringComparison.Ordinal))
            .Any(isExpected => !isExpected))
        {
            return false;
        }

        if (returnedTuple.Items
            .Select((item, index) => item is CompilerReferenceExpression reference
                && string.Equals(reference.Name, declarationStatements[index].Name, StringComparison.Ordinal))
            .Any(isExpected => !isExpected))
        {
            return false;
        }

        if (indexers.Skip(1).Any(indexer => !EqualityComparer<CompilerExpression>.Default.Equals(indexer.Target, indexers[0].Target)))
        {
            return false;
        }

        invocation = (CompilerInvocationExpression)indexers[0].Target;
        return true;
    }

    private CompilerComputationTreeBuilder CreateBuilder(
        MethodDeclarationSyntax method,
        CompilerTorchSharpModuleDescriptor? descriptor
    )
    {
        var name = descriptor?.TypeName ?? method.Identifier.ValueText;
        var builder = new CompilerComputationTreeBuilder(name);
        builder.SetDocument(_document);
        return builder;
    }

    private void RegisterValueContracts(
        CompilerComputationTreeBuilder builder,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs
    )
    {
        foreach (var input in inputs)
        {
            var inputValue = new CompilerValue(
                input.Name,
                input.Type,
                input.NameNode is null ? null : Span(input.NameNode));
            builder.AddInput(inputValue);
        }

        foreach (var output in outputs)
        {
            var outputValue = new CompilerValue(output.Name, output.Type);
            builder.AddOutput(outputValue);
        }
    }

    private static void RegisterStateMetadata(
        CompilerComputationTreeBuilder builder,
        CompilerTorchSharpModuleDescriptor? descriptor
    )
    {
        if (descriptor is null)
        {
            return;
        }

        foreach (var member in descriptor.StateMembers)
        {
            var stateMember = new CompilerStateMember(
                name: member.Name,
                kind: member.Kind,
                type: member.Type,
                value: member.Value);
            builder.AddStateMember(stateMember);
            builder.AddMetadata($"state:{member.Name}", member.CSharpTypeName);
        }

        foreach (var child in descriptor.ChildModules)
        {
            builder.AddMetadata($"child:{child.Name}", child.TypeName);
        }

        foreach (var helper in descriptor.HelperMethods)
        {
            builder.AddMetadata($"helper:{helper.Name}", "true");
        }
    }

    private IReadOnlyDictionary<string, CompilerComputationBlock> AddHelperBlocks(
        CompilerComputationTreeBuilder builder,
        IEnumerable<MethodDeclarationSyntax>? helperMethods
    )
    {
        var result = new Dictionary<string, CompilerComputationBlock>(StringComparer.Ordinal);
        foreach (var helper in helperMethods ?? Array.Empty<MethodDeclarationSyntax>())
        {
            ValidateNoRecursiveCall(helper);
            ValidateNoDynamicModuleDispatch(helper);
            var helperInputs = GetInputs(helper, descriptor: null);
            var helperOutputs = GetOutputs(helper, descriptor: null);
            var block = new CompilerComputationBlock(
                name: helper.Identifier.ValueText,
                inputs: helperInputs.Select(x => new CompilerValueReference(x.Name)),
                outputs: helperOutputs.Select(x => new CompilerValueReference(x.Name)),
                body: ScanBody(helper),
                span: Span(helper));
            builder.AddBlock(block);
            result.Add(block.Name, block);
            for (var index = 0; index < helperInputs.Count; index++)
            {
                builder.AddMetadata(
                    $"block-signature:{block.Name}:input:{index}",
                    helperInputs[index].CSharpTypeName ?? CSharpTypeName(helperInputs[index].Type));
            }

            for (var index = 0; index < helperOutputs.Count; index++)
            {
                builder.AddMetadata(
                    $"block-signature:{block.Name}:output:{index}",
                    helperOutputs[index].CSharpTypeName ?? CSharpTypeName(helperOutputs[index].Type));
            }
        }

        return result;
    }

    private static (IReadOnlyList<ScanValue> Inputs, IReadOnlyList<ScanValue> Outputs) ApplyOperatorTypeContracts(
        CompilerBlockStatement body,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs
    )
    {
        if (!TryGetStaticReturnExpression(body, out var returnExpression)
            || returnExpression is not CompilerInvocationExpression invocation
            || !CompilerOperatorMappingRegistry.TryGetTorchSharpCall(invocation.Target, out var mapping, out var receiver)
            || mapping is null)
        {
            return (inputs, outputs);
        }

        var inputExpressions = new List<CompilerExpression>();
        if (receiver is not null)
        {
            inputExpressions.Add(receiver);
        }

        inputExpressions.AddRange(invocation.Arguments);
        var typedInputs = inputs.ToDictionary(input => input.Name, input => input, StringComparer.Ordinal);
        for (var index = 0; index < Math.Min(inputExpressions.Count, mapping.InputElementTypes.Count); index++)
        {
            if (mapping.InputElementTypes[index] is not { } elementType
                || inputExpressions[index] is not CompilerReferenceExpression reference
                || !typedInputs.TryGetValue(reference.Name, out var input)
                || input.Type is not CompilerTensorType tensorType)
            {
                continue;
            }

            typedInputs[reference.Name] = new ScanValue(
                input.Name,
                new CompilerTensorType(elementType, tensorType.Dimensions, tensorType.Denotation),
                input.NameNode,
                input.CSharpTypeName);
        }

        var castOutputType = mapping.OnnxName == "Cast"
            && invocation.Arguments.Count == 1
            && TryGetMemberPath(invocation.Arguments[0], out var targetTypeName)
            && TryGetCastType(targetTypeName, out var castTypeCode)
                ? CompilerElementTypeFromOnnxCastCode(castTypeCode)
                : (CompilerElementType?)null;
        var outputElementType = castOutputType ?? mapping.OutputElementType;
        var typedOutputs = outputElementType is { } outputType && outputs.Count == 1
            ? [new ScanValue(
                outputs[0].Name,
                new CompilerTensorType(
                    outputType,
                    outputs[0].Type is CompilerTensorType outputTensorType ? outputTensorType.Dimensions : null),
                outputs[0].NameNode,
                outputs[0].CSharpTypeName)]
            : outputs;
        return (inputs.Select(input => typedInputs[input.Name]).ToArray(), typedOutputs);
    }

    private static bool TryGetCastType(string memberPath, out long onnxType)
    {
        var typeName = memberPath.Split('.').Last();
        (onnxType, _) = typeName switch
        {
            "Float32" or "Float" => (1L, true),
            "UInt8" or "Byte" => (2L, true),
            "Int8" or "SByte" => (3L, true),
            "UInt16" => (4L, true),
            "Int16" or "Short" => (5L, true),
            "Int32" or "Int" => (6L, true),
            "Int64" or "Long" => (7L, true),
            "Bool" or "Boolean" => (9L, true),
            "Float16" or "Half" => (10L, true),
            "Float64" or "Double" => (11L, true),
            "UInt32" => (12L, true),
            "UInt64" => (13L, true),
            "BFloat16" => (16L, true),
            _ => (0L, false),
        };
        return onnxType != 0;
    }

    private static bool TryGetMemberPath(CompilerExpression expression, out string path)
    {
        switch (expression)
        {
            case CompilerReferenceExpression reference:
                path = reference.Name;
                return true;
            case CompilerMemberAccessExpression member when TryGetMemberPath(member.Target, out var prefix):
                path = $"{prefix}.{member.MemberName}";
                return true;
            default:
                path = string.Empty;
                return false;
        }
    }

    private static CompilerElementType CompilerElementTypeFromOnnxCastCode(long onnxType) => onnxType switch
    {
        1 => CompilerElementType.Float32,
        2 => CompilerElementType.UInt8,
        3 => CompilerElementType.Int8,
        4 => CompilerElementType.UInt16,
        5 => CompilerElementType.Int16,
        6 => CompilerElementType.Int32,
        7 => CompilerElementType.Int64,
        9 => CompilerElementType.Boolean,
        10 => CompilerElementType.Float16,
        11 => CompilerElementType.Float64,
        12 => CompilerElementType.UInt32,
        13 => CompilerElementType.UInt64,
        16 => CompilerElementType.BFloat16,
        _ => CompilerElementType.Unknown,
    };

    private static string CSharpTypeName(CompilerType type)
    {
        return type switch
        {
            CompilerTensorType => "global::TorchSharp.torch.Tensor",
            CompilerScalarType scalar => scalar.ElementType switch
            {
                CompilerElementType.Boolean => "bool",
                CompilerElementType.Int32 => "int",
                CompilerElementType.Int64 => "long",
                CompilerElementType.Float32 => "float",
                CompilerElementType.Float64 => "double",
                _ => "object",
            },
            _ => "object",
        };
    }

    private static bool TryGetReferenceName(CompilerExpression expression, out string name)
    {
        if (expression is CompilerReferenceExpression reference)
        {
            name = reference.Name;
            return true;
        }

        name = string.Empty;
        return false;
    }

    private void ValidateNoRecursiveCall(MethodDeclarationSyntax method)
    {
        var methodName = method.Identifier.ValueText;
        var recursiveCall = method.DescendantNodes()
            .OfType<InvocationExpressionSyntax>()
            .FirstOrDefault(invocation => invocation.Expression switch
            {
                IdentifierNameSyntax identifier => string.Equals(identifier.Identifier.ValueText, methodName, StringComparison.Ordinal),
                MemberAccessExpressionSyntax
                {
                    Expression: ThisExpressionSyntax,
                    Name: var name,
                } => string.Equals(name.Identifier.ValueText, methodName, StringComparison.Ordinal),
                _ => false,
            });
        if (recursiveCall is not null)
        {
            throw Unsupported(method, $"Recursive call to '{methodName}' is not supported by the compiler.");
        }
    }

    private void ValidateNoDynamicModuleDispatch(MethodDeclarationSyntax method)
    {
        var dynamicDispatch = method.DescendantNodes()
            .OfType<InvocationExpressionSyntax>()
            .FirstOrDefault(invocation => invocation.Expression is MemberAccessExpressionSyntax member
                && string.Equals(member.Name.Identifier.ValueText, "forward", StringComparison.Ordinal));
        if (dynamicDispatch is not null)
        {
            throw Unsupported(dynamicDispatch, "Dynamic module forward dispatch is not supported.");
        }
    }

    private static string GetInvocationName(ExpressionSyntax expression)
    {
        return expression switch
        {
            IdentifierNameSyntax identifier => identifier.Identifier.ValueText,
            MemberAccessExpressionSyntax member => member.Name.Identifier.ValueText,
            _ => string.Empty,
        };
    }

    private static bool TryGetStaticReturnExpression(
        CompilerStatement statement,
        out CompilerExpression? expression
    )
    {
        switch (statement)
        {
            case CompilerReturnStatement { Expression: not null } returnStatement:
                expression = returnStatement.Expression;
                return true;
            case CompilerBlockStatement block when block.Statements.Count == 1:
                return TryGetStaticReturnExpression(block.Statements[0], out expression);
            case CompilerStaticIfStatement conditional
                when TryEvaluateStaticBoolean(conditional.Condition, out var condition):
                var selected = condition
                    ? conditional.WhenTrue
                    : conditional.WhenFalse ?? new CompilerBlockStatement([]);
                return TryGetStaticReturnExpression(selected, out expression);
            default:
                expression = null;
                return false;
        }
    }

    private static bool TryEvaluateStaticBoolean(CompilerExpression expression, out bool value)
    {
        if (expression is CompilerLiteralExpression { Literal: CompilerBooleanLiteral literal })
        {
            value = literal.Value;
            return true;
        }

        if (expression is CompilerUnaryExpression { Operator: "!", Expression: var operand }
            && TryEvaluateStaticBoolean(operand, out var operandValue))
        {
            value = !operandValue;
            return true;
        }

        value = false;
        return false;
    }

    private IReadOnlyList<ScanValue> GetInputs(
        MethodDeclarationSyntax method,
        CompilerTorchSharpModuleDescriptor? descriptor
    )
    {
        var contracts = descriptor?.Inputs ?? Array.Empty<CompilerTorchSharpValueDescriptor>();
        var result = method.ParameterList.Parameters
            .Select(
                (parameter, index) =>
                {
                    var contract = index < contracts.Count ? contracts[index] : null;
                    return new ScanValue(
                        parameter.Identifier.ValueText,
                        contract?.Type ?? InferType(parameter.Type?.ToString()),
                        parameter,
                        contract?.CSharpTypeName ?? parameter.Type?.ToString());
                })
            .ToArray();
        return result;
    }

    private IReadOnlyList<ScanValue> GetOutputs(
        MethodDeclarationSyntax method,
        CompilerTorchSharpModuleDescriptor? descriptor
    )
    {
        if (descriptor?.Outputs is { Count: > 0 })
        {
            return descriptor.Outputs
                .Select(x => new ScanValue(x.Name, x.Type, null, x.CSharpTypeName))
                .ToArray();
        }

        var returnType = method.ReturnType.ToString();
        if (returnType.StartsWith("(", StringComparison.Ordinal))
        {
            var parts = returnType
                .Trim('(', ')')
                .Split(',')
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .ToArray();
            return Enumerable.Range(0, parts.Length)
                .Select(index => new ScanValue($"output{index}", InferType(parts[index]), null, parts[index]))
                .ToArray();
        }

        return [new ScanValue("output", InferType(returnType), null, returnType)];
    }

    private CompilerBlockStatement ScanBody(MethodDeclarationSyntax method)
    {
        if (method.Body is not null)
        {
            return new CompilerBlockStatement(method.Body.Statements.Select(ScanStatement));
        }

        if (method.ExpressionBody is not null)
        {
            return new CompilerBlockStatement(
            [
                new CompilerReturnStatement(
                    ScanExpression(method.ExpressionBody.Expression),
                    Span(method.ExpressionBody)),
            ]);
        }

        throw Unsupported(method, "A method must have a body or expression body.");
    }

    /// <summary>
    /// Преобразует C# statement наиболее конкретным syntax-путём и сразу отклоняет динамические конструкции.
    /// Такое dispatch сохраняет порядок statements и не маскирует неподдерживаемый код как opaque node.
    /// </summary>
    private CompilerStatement ScanStatement(StatementSyntax statement)
    {
        switch (statement)
        {
            case BlockSyntax block:
                return new CompilerBlockStatement(block.Statements.Select(ScanStatement), Span(block));

            case LocalDeclarationStatementSyntax declaration:
                return ScanLocalDeclaration(declaration);

            case ExpressionStatementSyntax
            {
                Expression: AssignmentExpressionSyntax assignment,
            } expressionStatement:
                if (assignment.Left is DeclarationExpressionSyntax declarationExpression)
                {
                    return ScanDeconstructionAssignment(
                        declarationExpression.Designation,
                        assignment.Right,
                        expressionStatement);
                }

                return new CompilerAssignmentStatement(
                    target: ScanExpression(assignment.Left),
                    value: ScanExpression(assignment.Right),
                    span: Span(expressionStatement),
                    @operator: assignment.OperatorToken.ValueText);

            case ExpressionStatementSyntax expressionStatement:
                return new CompilerExpressionStatement(
                    ScanExpression(expressionStatement.Expression),
                    Span(expressionStatement));

            case ReturnStatementSyntax returnStatement:
                return new CompilerReturnStatement(
                    returnStatement.Expression is null ? null : ScanExpression(returnStatement.Expression),
                    Span(returnStatement));

            case IfStatementSyntax ifStatement:
                if (!TryEvaluateStaticBoolean(ifStatement.Condition, out _))
                {
                    throw Unsupported(
                        ifStatement,
                        "Only statically resolvable if conditions are supported by the compiler frontend.");
                }

                return new CompilerStaticIfStatement(
                    condition: ScanExpression(ifStatement.Condition),
                    whenTrue: ScanStatement(ifStatement.Statement),
                    whenFalse: ifStatement.Else is null ? null : ScanStatement(ifStatement.Else.Statement),
                    span: Span(ifStatement));

            case ForEachStatementSyntax foreachStatement:
                if (!IsStaticCollection(foreachStatement))
                {
                    throw Unsupported(
                        foreachStatement,
                        "Only foreach loops over compile-time array literals or local arrays are supported.");
                }

                return new CompilerStaticForeachStatement(
                    variableName: foreachStatement.Identifier.ValueText,
                    collection: ScanExpression(foreachStatement.Expression),
                    body: ScanStatement(foreachStatement.Statement),
                    span: Span(foreachStatement));

            case UsingStatementSyntax usingStatement:
                return ScanUsingStatement(usingStatement);

            case EmptyStatementSyntax empty:
                return new CompilerBlockStatement(Array.Empty<CompilerStatement>(), Span(empty));

            default:
                throw Unsupported(
                    statement,
                    $"C# statement '{statement.Kind()}' is not supported by the compiler frontend.");
        }
    }

    private CompilerStatement ScanDeconstructionAssignment(
        VariableDesignationSyntax designation,
        ExpressionSyntax value,
        SyntaxNode source
    )
    {
        if (designation is not ParenthesizedVariableDesignationSyntax tupleDesignation)
        {
            throw Unsupported(source, "Only tuple deconstruction into named locals is supported.");
        }

        var statements = new List<CompilerStatement>();
        for (var index = 0; index < tupleDesignation.Variables.Count; index++)
        {
            if (tupleDesignation.Variables[index] is not SingleVariableDesignationSyntax variable
                || string.IsNullOrWhiteSpace(variable.Identifier.ValueText)
                || variable.Identifier.ValueText == "_")
            {
                continue;
            }

            statements.Add(new CompilerDeclarationStatement(
                name: variable.Identifier.ValueText,
                initializer: new CompilerMemberAccessExpression(
                    ScanExpression(value),
                    $"Item{index + 1}",
                    Span(variable)),
                span: Span(variable)));
        }

        return new CompilerBlockStatement(statements, Span(source));
    }

    private static bool IsStaticCollection(ForEachStatementSyntax foreachStatement)
    {
        if (TryGetArrayElements(foreachStatement.Expression, out var elements))
        {
            return elements.All(IsCompileTimeLiteral);
        }

        if (foreachStatement.Expression is not IdentifierNameSyntax identifier)
        {
            return false;
        }

        var method = foreachStatement.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();
        var declaration = method?.DescendantNodes()
            .OfType<VariableDeclaratorSyntax>()
            .Where(variable => variable.SpanStart < foreachStatement.SpanStart)
            .Where(variable => string.Equals(variable.Identifier.ValueText, identifier.Identifier.ValueText, StringComparison.Ordinal))
            .OrderByDescending(variable => variable.SpanStart)
            .FirstOrDefault();
        return declaration?.Initializer?.Value is { } initializer
            && TryGetArrayElements(initializer, out elements)
            && elements.All(IsCompileTimeLiteral);
    }

    private static bool TryGetArrayElements(ExpressionSyntax expression, out SeparatedSyntaxList<ExpressionSyntax> elements)
    {
        var initializer = expression switch
        {
            ArrayCreationExpressionSyntax arrayCreation => arrayCreation.Initializer,
            ImplicitArrayCreationExpressionSyntax implicitArrayCreation => implicitArrayCreation.Initializer,
            InitializerExpressionSyntax initializerExpression => initializerExpression,
            _ => null,
        };
        if (initializer is null)
        {
            elements = default;
            return false;
        }

        elements = initializer.Expressions;
        return true;
    }

    private static bool IsCompileTimeLiteral(ExpressionSyntax expression)
    {
        return expression is LiteralExpressionSyntax
            || expression is PrefixUnaryExpressionSyntax prefix
                && prefix.IsKind(SyntaxKind.UnaryMinusExpression)
                && prefix.Operand is LiteralExpressionSyntax;
    }

    private CompilerStatement ScanUsingStatement(UsingStatementSyntax usingStatement)
    {
        var statements = new List<CompilerStatement>();
        if (usingStatement.Declaration is not null)
        {
            if (usingStatement.Declaration.Variables.Any(
                x => x.Initializer?.Value is not null
                    && !IsNoOpUsingExpression(x.Initializer.Value)))
            {
                statements.Add(ScanVariableDeclaration(usingStatement.Declaration, usingStatement.Declaration));
            }
        }
        else if (usingStatement.Expression is not null
            && !IsNoOpUsingExpression(usingStatement.Expression))
        {
            statements.Add(
                new CompilerExpressionStatement(
                    ScanExpression(usingStatement.Expression),
                    Span(usingStatement.Expression)));
        }

        statements.Add(ScanStatement(usingStatement.Statement));
        return new CompilerBlockStatement(statements, Span(usingStatement));
    }

    private static bool IsNoOpUsingExpression(ExpressionSyntax expression)
    {
        return expression is InvocationExpressionSyntax
        {
            Expression: MemberAccessExpressionSyntax member,
        } && string.Equals(member.Name.Identifier.ValueText, "no_grad", StringComparison.Ordinal);
    }

    private CompilerStatement ScanLocalDeclaration(LocalDeclarationStatementSyntax declaration)
    {
        return ScanVariableDeclaration(declaration.Declaration, declaration);
    }

    private CompilerStatement ScanVariableDeclaration(
        VariableDeclarationSyntax declaration,
        SyntaxNode source
    )
    {
        var statements = new List<CompilerStatement>();
        foreach (var variable in declaration.Variables)
        {
            var variableName = variable.Identifier.ValueText;
            if (string.IsNullOrWhiteSpace(variableName))
            {
                if (variable.Initializer?.Value is null)
                {
                    throw Unsupported(variable, "A deconstruction declaration requires an initializer.");
                }

                var variableText = variable.ToString();
                var assignmentIndex = variableText.IndexOf('=');
                var names = (assignmentIndex >= 0 ? variableText.Substring(0, assignmentIndex) : variableText)
                    .Trim()
                    .Trim('(', ')')
                    .Split(',')
                    .Select(x => x.Trim())
                    .ToArray();
                for (var index = 0; index < names.Length; index++)
                {
                    var name = names[index];
                    if (string.IsNullOrWhiteSpace(name) || name == "_")
                    {
                        continue;
                    }

                    statements.Add(
                        new CompilerDeclarationStatement(
                            name,
                        new CompilerMemberAccessExpression(
                                ScanExpression(variable.Initializer.Value),
                            $"Item{index + 1}",
                                Span(variable)),
                            Span(variable)));
                }

                continue;
            }

            statements.Add(
                new CompilerDeclarationStatement(
                    variableName,
                    variable.Initializer?.Value is null
                        ? null
                        : ScanExpression(variable.Initializer.Value),
                    Span(variable)));
        }

        var result = statements.Count == 1
            ? statements[0]
            : new CompilerBlockStatement(statements, Span(source));
        return result;
    }

    /// <summary>
    /// Преобразует поддержанные C# expressions в compiler-owned представление C# syntax с исходными spans.
    /// Нераспознанные формы завершаются диагностикой на этапе Analyze, а не теряются при генерации.
    /// </summary>
    private CompilerExpression ScanExpression(ExpressionSyntax expression)
    {
        switch (expression)
        {
            case IdentifierNameSyntax identifier:
                return new CompilerReferenceExpression(identifier.Identifier.ValueText, Span(identifier));

            case ThisExpressionSyntax:
                return new CompilerReferenceExpression("this", Span(expression));

            case MemberAccessExpressionSyntax member:
                return new CompilerMemberAccessExpression(
                    ScanExpression(member.Expression),
                    member.Name.Identifier.ValueText,
                    Span(member));

            case InvocationExpressionSyntax invocation:
                return new CompilerInvocationExpression(
                    ScanExpression(invocation.Expression),
                    ScanInvocationArguments(invocation),
                    Span(invocation));

            case ElementAccessExpressionSyntax indexer:
                if (indexer.ArgumentList.Arguments.Count != 1)
                {
                    throw Unsupported(indexer, "Only one-dimensional indexers are supported.");
                }

                return new CompilerIndexerExpression(
                    ScanExpression(indexer.Expression),
                    ScanExpression(indexer.ArgumentList.Arguments[0].Expression),
                    Span(indexer));

            case ParenthesizedExpressionSyntax parenthesized:
                return ScanExpression(parenthesized.Expression);

            case CastExpressionSyntax cast:
                return ScanExpression(cast.Expression);

            case TupleExpressionSyntax tuple:
                return new CompilerTupleExpression(
                    tuple.Arguments.Select(x => ScanExpression(x.Expression)),
                    Span(tuple));

            case ArrayCreationExpressionSyntax array:
                return ScanArrayInitializer(array.Initializer, array);

            case ImplicitArrayCreationExpressionSyntax implicitArray:
                return ScanArrayInitializer(implicitArray.Initializer, implicitArray);

            case InitializerExpressionSyntax initializer:
                return new CompilerArrayExpression(
                    initializer.Expressions.Select(ScanExpression),
                    Span(initializer));

            case LiteralExpressionSyntax literal:
                return new CompilerLiteralExpression(ScanLiteral(literal), Span(literal));

            case BinaryExpressionSyntax binary:
                return new CompilerBinaryExpression(
                    left: ScanExpression(binary.Left),
                    @operator: binary.OperatorToken.ValueText,
                    right: ScanExpression(binary.Right),
                    span: Span(binary));

            case PrefixUnaryExpressionSyntax prefix:
                return new CompilerUnaryExpression(
                    prefix.OperatorToken.ValueText,
                    ScanExpression(prefix.Operand),
                    Span(prefix));

            case PostfixUnaryExpressionSyntax postfix:
                return new CompilerUnaryExpression(
                    postfix.OperatorToken.ValueText,
                    ScanExpression(postfix.Operand),
                    Span(postfix));

            case ConditionalExpressionSyntax conditional:
                if (!TryEvaluateStaticBoolean(conditional.Condition, out var condition))
                {
                    throw Unsupported(conditional, "Only statically resolvable conditional expressions are supported.");
                }

                var result = ScanExpression(condition ? conditional.WhenTrue : conditional.WhenFalse);
                return result;

            default:
                throw Unsupported(
                    expression,
                    $"C# expression '{expression.Kind()}' is not supported by the compiler frontend.");
        }
    }

    private IEnumerable<CompilerExpression> ScanInvocationArguments(InvocationExpressionSyntax invocation)
    {
        var arguments = invocation.ArgumentList.Arguments;
        if (!TryGetSyntaxMemberPath(invocation.Expression, out var targetPath)
            || !targetPath.EndsWith("addmm", StringComparison.Ordinal)
            || !arguments.Any(static argument => argument.NameColon is not null))
        {
            return arguments.Select(argument => ScanExpression(argument.Expression)).ToArray();
        }

        var isStaticAddmm = targetPath.StartsWith("torch.", StringComparison.Ordinal)
            || targetPath.Contains(".torch.addmm", StringComparison.Ordinal);
        var tensorArgumentCount = isStaticAddmm ? 3 : 2;
        var positional = arguments.Where(static argument => argument.NameColon is null).ToArray();
        var named = arguments
            .Where(static argument => argument.NameColon is not null)
            .ToDictionary(
                static argument => argument.NameColon!.Name.Identifier.ValueText,
                argument => ScanExpression(argument.Expression),
                StringComparer.Ordinal);
        if (positional.Length < tensorArgumentCount
            || named.Keys.Any(static name => name is not ("beta" or "alpha")))
        {
            throw Unsupported(invocation, "Named Gemm syntax supports positional tensor operands and only beta/alpha named arguments.");
        }

        var normalized = positional
            .Take(tensorArgumentCount)
            .Select(argument => ScanExpression(argument.Expression))
            .ToList();
        var positionalAttributes = positional.Skip(tensorArgumentCount)
            .Select(argument => ScanExpression(argument.Expression))
            .ToArray();
        var beta = named.TryGetValue("beta", out var namedBeta)
            ? namedBeta
            : positionalAttributes.Length > 0 ? positionalAttributes[0] : FloatLiteral(1f);
        var alpha = named.TryGetValue("alpha", out var namedAlpha)
            ? namedAlpha
            : positionalAttributes.Length > 1 ? positionalAttributes[1] : FloatLiteral(1f);
        normalized.Add(beta);
        normalized.Add(alpha);
        return normalized;
    }

    private static CompilerExpression FloatLiteral(float value) => new CompilerLiteralExpression(
        new CompilerFloatingPointLiteral(CompilerElementType.Float32, value));

    private static bool TryGetSyntaxMemberPath(ExpressionSyntax expression, out string path)
    {
        switch (expression)
        {
            case IdentifierNameSyntax identifier:
                path = identifier.Identifier.ValueText;
                return true;
            case MemberAccessExpressionSyntax member when TryGetSyntaxMemberPath(member.Expression, out var prefix):
                path = $"{prefix}.{member.Name.Identifier.ValueText}";
                return true;
            case AliasQualifiedNameSyntax aliasQualified:
                path = $"{aliasQualified.Alias.Identifier.ValueText}::{aliasQualified.Name.Identifier.ValueText}";
                return true;
            default:
                path = string.Empty;
                return false;
        }
    }

    private CompilerExpression ScanArrayInitializer(
        InitializerExpressionSyntax? initializer,
        SyntaxNode source
    )
    {
        if (initializer is null)
        {
            throw Unsupported(source, "An array initializer is required.");
        }

        var result = new CompilerArrayExpression(
            initializer.Expressions.Select(ScanExpression),
            Span(source));
        return result;
    }

    private static CompilerLiteral ScanLiteral(LiteralExpressionSyntax literal)
    {
        if (literal.IsKind(SyntaxKind.NullLiteralExpression))
        {
            return new CompilerNullLiteral();
        }

        if (literal.IsKind(SyntaxKind.TrueLiteralExpression))
        {
            return new CompilerBooleanLiteral(true);
        }

        if (literal.IsKind(SyntaxKind.FalseLiteralExpression))
        {
            return new CompilerBooleanLiteral(false);
        }

        if (literal.IsKind(SyntaxKind.StringLiteralExpression)
            || literal.IsKind(SyntaxKind.CharacterLiteralExpression))
        {
            return new CompilerStringLiteral(literal.Token.ValueText);
        }

        return literal.Token.Value switch
        {
            byte value => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt8, value),
            sbyte value => new CompilerSignedIntegerLiteral(CompilerElementType.Int8, value),
            short value => new CompilerSignedIntegerLiteral(CompilerElementType.Int16, value),
            ushort value => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt16, value),
            int value => new CompilerSignedIntegerLiteral(CompilerElementType.Int32, value),
            uint value => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt32, value),
            long value => new CompilerSignedIntegerLiteral(CompilerElementType.Int64, value),
            ulong value => new CompilerUnsignedIntegerLiteral(CompilerElementType.UInt64, value),
            float value => new CompilerFloatingPointLiteral(CompilerElementType.Float32, value),
            double value => new CompilerFloatingPointLiteral(CompilerElementType.Float64, value),
            decimal value => new CompilerFloatingPointLiteral(CompilerElementType.Float64, (double)value),
            _ => throw new InvalidOperationException($"Unsupported literal '{literal.Token.ValueText}'."),
        };
    }

    private static bool TryEvaluateStaticBoolean(ExpressionSyntax expression, out bool value)
    {
        switch (expression)
        {
            case LiteralExpressionSyntax literal when literal.IsKind(SyntaxKind.TrueLiteralExpression):
                value = true;
                return true;
            case LiteralExpressionSyntax literal when literal.IsKind(SyntaxKind.FalseLiteralExpression):
                value = false;
                return true;
            case ParenthesizedExpressionSyntax parenthesized:
                return TryEvaluateStaticBoolean(parenthesized.Expression, out value);
            case PrefixUnaryExpressionSyntax prefix when prefix.IsKind(SyntaxKind.LogicalNotExpression):
                if (TryEvaluateStaticBoolean(prefix.Operand, out var operand))
                {
                    value = !operand;
                    return true;
                }

                break;
            case BinaryExpressionSyntax binary when binary.IsKind(SyntaxKind.LogicalAndExpression):
                if (TryEvaluateStaticBoolean(binary.Left, out var left)
                    && TryEvaluateStaticBoolean(binary.Right, out var right))
                {
                    value = left && right;
                    return true;
                }

                break;
            case BinaryExpressionSyntax binary when binary.IsKind(SyntaxKind.LogicalOrExpression):
                if (TryEvaluateStaticBoolean(binary.Left, out var orLeft)
                    && TryEvaluateStaticBoolean(binary.Right, out var orRight))
                {
                    value = orLeft || orRight;
                    return true;
                }

                break;
        }

        value = false;
        return false;
    }

    private void AddDeclaredValues(
        CompilerComputationTreeBuilder builder,
        CompilerBlockStatement body,
        IReadOnlyList<ScanValue> inputs,
        IReadOnlyList<ScanValue> outputs,
        CompilerTorchSharpModuleDescriptor? descriptor
    )
    {
        var existing = new HashSet<string>(
            inputs.Select(x => x.Name)
                .Concat(outputs.Select(x => x.Name))
                .Concat(descriptor?.StateMembers.Select(x => x.Name) ?? Array.Empty<string>()),
            StringComparer.Ordinal);

        foreach (var declaration in FindDeclarations(body))
        {
            if (existing.Add(declaration.Name))
            {
                builder.AddIntermediateValue(new CompilerValue(declaration.Name, DEFAULT_TENSOR_TYPE, declaration.Span));
            }
        }
    }

    private static IEnumerable<CompilerDeclarationStatement> FindDeclarations(CompilerStatement statement)
    {
        switch (statement)
        {
            case CompilerDeclarationStatement declaration:
                yield return declaration;
                break;
            case CompilerBlockStatement block:
                foreach (var child in block.Statements.SelectMany(FindDeclarations))
                {
                    yield return child;
                }

                break;
            case CompilerStaticIfStatement conditional:
                foreach (var child in FindDeclarations(conditional.WhenTrue))
                {
                    yield return child;
                }

                if (conditional.WhenFalse is not null)
                {
                    foreach (var child in FindDeclarations(conditional.WhenFalse))
                    {
                        yield return child;
                    }
                }

                break;
            case CompilerStaticForeachStatement loop:
                foreach (var child in FindDeclarations(loop.Body))
                {
                    yield return child;
                }

                break;
        }
    }

    private CompilerSourceSpan Span(SyntaxNode node) => CSharpCompilerFrontend.Span(_document, node.GetLocation());

    private CSharpCompilerDiagnosticException Unsupported(SyntaxNode node, string message)
    {
        return new CSharpCompilerDiagnosticException(
            code: CompilerDiagnosticCodes.Unsupported,
            message: message,
            stage: CompilerDiagnosticStage.Analyze,
            span: Span(node),
            context: new CompilerDiagnosticContext(
                caller: node.AncestorsAndSelf()
                    .OfType<MethodDeclarationSyntax>()
                    .FirstOrDefault()?.Identifier.ValueText
                    ?? _context?.Descriptor.MethodName
                    ?? "forward",
                callee: node.Kind().ToString()));
    }

    private static CompilerType InferType(string? csharpTypeName)
    {
        if (string.IsNullOrWhiteSpace(csharpTypeName)
            || csharpTypeName.Contains("Tensor", StringComparison.Ordinal))
        {
            return DEFAULT_TENSOR_TYPE;
        }

        return csharpTypeName switch
        {
            "bool" => new CompilerScalarType(CompilerElementType.Boolean),
            "byte" => new CompilerScalarType(CompilerElementType.UInt8),
            "sbyte" => new CompilerScalarType(CompilerElementType.Int8),
            "short" => new CompilerScalarType(CompilerElementType.Int16),
            "ushort" => new CompilerScalarType(CompilerElementType.UInt16),
            "int" => new CompilerScalarType(CompilerElementType.Int32),
            "uint" => new CompilerScalarType(CompilerElementType.UInt32),
            "long" => new CompilerScalarType(CompilerElementType.Int64),
            "ulong" => new CompilerScalarType(CompilerElementType.UInt64),
            "float" => new CompilerScalarType(CompilerElementType.Float32),
            "double" => new CompilerScalarType(CompilerElementType.Float64),
            "string" => new CompilerScalarType(CompilerElementType.String),
            _ => new CompilerOpaqueType("CSharp", csharpTypeName ?? string.Empty),
        };
    }

    private sealed class ScanValue
    {
        public ScanValue(
            string name,
            CompilerType type,
            SyntaxNode? nameNode,
            string? csharpTypeName)
        {
            Name = name;
            Type = type;
            NameNode = nameNode;
            CSharpTypeName = csharpTypeName;
        }

        public string Name { get; }

        public CompilerType Type { get; }

        public SyntaxNode? NameNode { get; }

        public string? CSharpTypeName { get; }
    }
}

internal sealed class CSharpCompilerDiagnosticException : Exception
{
    public CSharpCompilerDiagnosticException(
        string code,
        string message,
        CompilerDiagnosticStage stage,
        CompilerSourceSpan? span = null,
        CompilerDiagnosticContext? context = null
    ) : base(message)
    {
        Code = code;
        Stage = stage;
        Span = span;
        Context = context;
    }

    public string Code { get; }

    public CompilerDiagnosticStage Stage { get; }

    public CompilerSourceSpan? Span { get; }

    public CompilerDiagnosticContext? Context { get; }
}

internal static class CSharpCompilerBackend
{
    /// <summary>
    /// Генерирует исходный TorchSharp-класс из представления C# syntax и возвращает diagnostics для конструкций без C# mapping.
    /// Проверка до печати не даёт случайно представить ONNX operations как исполняемый C# код.
    /// </summary>
    public static CompilerResult<string> Generate(
        CompilerComputationTree tree,
        CompilerCSharpGenerationOptions? options = null
    )
    {
        options ??= new CompilerCSharpGenerationOptions();
        try
        {
            ValidateOptions(options);
            if (tree.Operations.OfType<CompilerOperation>().Any(operation =>
                !CompilerOperatorMappingRegistry.TryGetOnnx(
                    operation.Descriptor.Domain,
                    operation.Descriptor.Name,
                    out var mapping)
                || mapping is null
                || !mapping.Accepts(operation)))
            {
                return CompilerResult<string>.Failure(
                [
                    new CompilerDiagnostic(
                        code: CompilerDiagnosticCodes.Unsupported,
                        message: "The compiler tree contains an operation without a registered TorchSharp import mapping.",
                        stage: CompilerDiagnosticStage.Emit,
                        severity: CompilerDiagnosticSeverity.Error),
                ]);
            }

            if (tree.SyntaxBody is null && tree.Operations.Count == 0)
            {
                return CompilerResult<string>.Failure(
                [
                    new CompilerDiagnostic(
                        code: CompilerDiagnosticCodes.Unsupported,
                        message: "The compiler tree has no C# syntax body to emit.",
                        stage: CompilerDiagnosticStage.Emit,
                        severity: CompilerDiagnosticSeverity.Error),
                ]);
            }

            var printer = new CSharpSourcePrinter(tree, options);
            var source = printer.Print();
            var result = CompilerResult<string>.Success(source);
            return result;
        }
        catch (CSharpCompilerDiagnosticException exception)
        {
            return CompilerResult<string>.Failure(
            [
                new CompilerDiagnostic(
                    code: exception.Code,
                    message: exception.Message,
                    stage: exception.Stage,
                    severity: CompilerDiagnosticSeverity.Error,
                    span: exception.Span,
                    context: exception.Context),
            ]);
        }
    }

    private static void ValidateOptions(CompilerCSharpGenerationOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.Namespace)
            || string.IsNullOrWhiteSpace(options.ClassName)
            || string.IsNullOrWhiteSpace(options.ModuleName))
        {
            throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: "C# generation options require namespace, class name, and module name.",
                stage: CompilerDiagnosticStage.Emit);
        }
    }
}

internal sealed class CSharpSourcePrinter
{
    private const string TORCH_TENSOR_TYPE = "global::TorchSharp.torch.Tensor";
    private const string TORCH_MODULE_TYPE = "global::TorchSharp.torch.nn.Module";

    private readonly CompilerComputationTree _tree;
    private readonly CompilerCSharpGenerationOptions _options;
    private readonly StringBuilder _builder = new();
    private int _indent;

    public CSharpSourcePrinter(
        CompilerComputationTree tree,
        CompilerCSharpGenerationOptions options
    )
    {
        _tree = tree;
        _options = options;
    }

    /// <summary>
    /// Собирает C# модуль из compiler-owned промежуточного представления, сохраняя порядок блоков и инструкций.
    /// Печать разделена по структуре класса, чтобы каждую часть генерации можно было менять независимо.
    /// </summary>
    public string Print()
    {
        PrintFileHeader();
        PrintModuleMembers();
        PrintForwardMethod();

        var result = _builder.ToString();
        return result;
    }

    private void PrintFileHeader()
    {
        if (_options.IncludeNullableContext)
        {
            Line("#nullable enable");
        }

        Line("using System;");
        Line("using global::TorchSharp;");
        Line("using static global::TorchSharp.torch;");
        Line();
        Line($"namespace {_options.Namespace};");
        Line();
        Line($"public sealed class {_options.ClassName} : {ModuleBaseType()}");
        Line("{");
        _indent++;
        Line($"public {_options.ClassName}(string name = \"{Escape(_options.ModuleName)}\") : base(name)");
        Line("{");
        _indent++;
        Line("}");
        _indent--;
    }

    private void PrintModuleMembers()
    {
        foreach (var member in _tree.Parameters.Concat(_tree.Buffers).Concat(_tree.Initializers))
        {
            var memberType = TypeName(member.Type);
            var memberValue = member.Value is null ? "default" : Literal(member.Value);
            Line($"private {memberType} {member.Name} = {memberValue};");
        }

        foreach (var child in _tree.Metadata.Where(x => x.Key.StartsWith("child:", StringComparison.Ordinal)))
        {
            var childName = child.Key.Substring("child:".Length);
            Line($"private {TORCH_MODULE_TYPE}<{TORCH_TENSOR_TYPE}, {TORCH_TENSOR_TYPE}> {childName} = default;");
        }

        foreach (var block in _tree.Blocks)
        {
            Line();
            PrintBlock(block);
        }
    }

    private void PrintForwardMethod()
    {
        Line();
        Line($"public override {OutputType()} forward({InputParameters()})");
        Line("{");
        _indent++;
        var hasReturn = false;
        if (_tree.SyntaxBody is not null)
        {
            PrintStatement(_tree.SyntaxBody);
            hasReturn = AlwaysReturns(_tree.SyntaxBody);
        }

        if (_tree.Operations.Count != 0)
        {
            hasReturn = PrintOperations() || hasReturn;
        }

        if (!hasReturn)
        {
            Line("return default;");
        }

        _indent--;
        Line("}");
        _indent--;
        Line("}");
    }

    private bool PrintOperations()
    {
        foreach (var operation in _tree.Operations)
        {
            if (operation is CompilerOperation mappedOperation)
            {
                if (!CompilerOperatorMappingRegistry.TryGetOnnx(
                    mappedOperation.Descriptor.Domain,
                    mappedOperation.Descriptor.Name,
                    out var mapping)
                    || mapping is null
                    || mapping.Capability is not (CompilerOperationCapability.Bidirectional or CompilerOperationCapability.ImportOnly)
                    || !mapping.Accepts(mappedOperation))
                {
                    throw new CSharpCompilerDiagnosticException(
                        code: CompilerDiagnosticCodes.Unsupported,
                        message: $"Operation '{mappedOperation.Name}' has no TorchSharp import mapping.",
                        stage: CompilerDiagnosticStage.Emit,
                        span: mappedOperation.Span);
                }

                Line($"var {mappedOperation.Outputs[0].Name} = {mapping.EmitTorchSharpExpression(mappedOperation)};");
                continue;
            }

            if (operation is not CompilerModuleCall call)
            {
                throw new CSharpCompilerDiagnosticException(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"Operation '{operation.Name}' cannot be emitted as C# TorchSharp code.",
                    stage: CompilerDiagnosticStage.Emit,
                    span: operation.Span);
            }

            var inputs = string.Join(", ", call.Arguments.Select(Expression));
            if (call.Outputs.Count == 1 && !call.Outputs[0].IsEmptyOptional)
            {
                Line($"var {call.Outputs[0].Name} = {call.TargetBlock}({inputs});");
            }
            else
            {
                var outputNames = string.Join(", ", call.Outputs.Where(x => !x.IsEmptyOptional).Select(x => x.Name));
                Line($"var ({outputNames}) = {call.TargetBlock}({inputs});");
            }
        }

        var finalOutputs = _tree.Outputs.Select(x => x.Name).ToArray();
        if (finalOutputs.Length == 0)
        {
            return false;
        }

        var returnValue = finalOutputs.Length == 1
            ? finalOutputs[0]
            : $"({string.Join(", ", finalOutputs)})";
        Line($"return {returnValue};");

        return true;
    }

    private string ModuleBaseType()
    {
        var result = $"{TORCH_MODULE_TYPE}<{string.Join(", ", _tree.Inputs.Select(_ => TORCH_TENSOR_TYPE).Concat([OutputType()]))}>";
        return result;
    }

    private string InputParameters()
    {
        var result = string.Join(", ", _tree.Inputs.Select(x => $"{TORCH_TENSOR_TYPE} {x.Name}"));
        return result;
    }

    private string OutputType()
    {
        if (_tree.Outputs.Count == 1)
        {
            return TypeName(_tree.Outputs[0].Type);
        }

        var result = $"({string.Join(", ", _tree.Outputs.Select(x => $"{TypeName(x.Type)} {x.Name}"))})";
        return result;
    }

    private static string TypeName(CompilerType type)
    {
        return type switch
        {
            CompilerTensorType => TORCH_TENSOR_TYPE,
            CompilerScalarType scalar => scalar.ElementType switch
            {
                CompilerElementType.Boolean => "bool",
                CompilerElementType.Int8 => "sbyte",
                CompilerElementType.UInt8 => "byte",
                CompilerElementType.Int16 => "short",
                CompilerElementType.UInt16 => "ushort",
                CompilerElementType.Int32 => "int",
                CompilerElementType.UInt32 => "uint",
                CompilerElementType.Int64 => "long",
                CompilerElementType.UInt64 => "ulong",
                CompilerElementType.Float32 => "float",
                CompilerElementType.Float64 => "double",
                CompilerElementType.String => "string",
                _ => "object",
            },
            CompilerTupleType tuple => $"({string.Join(", ", tuple.ElementTypes.Select(TypeName))})",
            _ => "object",
        };
    }

    private void PrintBlock(CompilerComputationBlock block)
    {
        var outputTypes = block.Outputs
            .Select((output, index) => BlockValueType(block.Name, "output", index))
            .ToArray();
        var outputType = block.Outputs.Count == 1
            ? outputTypes[0]
            : $"({string.Join(", ", block.Outputs.Select((output, index) => $"{outputTypes[index]} {output.Name}"))})";
        var inputParameters = block.Inputs
            .Select((input, index) => $"{BlockValueType(block.Name, "input", index)} {input.Name}");
        Line($"private {outputType} {block.Name}({string.Join(", ", inputParameters)})");
        Line("{");
        _indent++;
        PrintStatement(block.Body);
        if (!AlwaysReturns(block.Body))
        {
            Line("return default;");
        }

        _indent--;
        Line("}");
    }

    private string BlockValueType(string blockName, string kind, int index)
    {
        var metadataKey = $"block-signature:{blockName}:{kind}:{index}";
        var declaredType = _tree.Metadata
            .FirstOrDefault(item => string.Equals(item.Key, metadataKey, StringComparison.Ordinal))
            .Value;
        return string.IsNullOrWhiteSpace(declaredType) ? TORCH_TENSOR_TYPE : declaredType;
    }

    private void PrintStatement(CompilerStatement statement)
    {
        switch (statement)
        {
            case CompilerBlockStatement block:
                foreach (var child in block.Statements)
                {
                    PrintStatement(child);
                }

                break;
            case CompilerDeclarationStatement declaration:
                Line($"var {declaration.Name}{(declaration.Initializer is null ? string.Empty : $" = {Expression(declaration.Initializer)}")};");
                break;
            case CompilerAssignmentStatement assignment:
                Line($"{Expression(assignment.Target)} {assignment.Operator} {Expression(assignment.Value)};");
                break;
            case CompilerReturnStatement @return:
                Line($"return{(@return.Expression is null ? string.Empty : $" {Expression(@return.Expression)}")};");
                break;
            case CompilerExpressionStatement expression:
                Line($"{Expression(expression.Expression)};");
                break;
            case CompilerStaticIfStatement conditional:
                Line($"if ({Expression(conditional.Condition)})");
                Line("{");
                _indent++;
                PrintStatement(conditional.WhenTrue);
                _indent--;
                Line("}");
                if (conditional.WhenFalse is not null)
                {
                    Line("else");
                    Line("{");
                    _indent++;
                    PrintStatement(conditional.WhenFalse);
                    _indent--;
                    Line("}");
                }

                break;
            case CompilerStaticForeachStatement loop:
                Line($"foreach (var {loop.VariableName} in {Expression(loop.Collection)})");
                Line("{");
                _indent++;
                PrintStatement(loop.Body);
                _indent--;
                Line("}");
                break;
            default:
                throw new CSharpCompilerDiagnosticException(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"C# statement '{statement.GetType().Name}' cannot be emitted.",
                    stage: CompilerDiagnosticStage.Emit,
                    span: statement.Span);
        }
    }

    private string Expression(CompilerExpression expression)
    {
        return expression switch
        {
            CompilerReferenceExpression reference => reference.Name,
            CompilerLiteralExpression literal => Literal(literal.Literal),
            CompilerArrayExpression array => $"new[] {{ {string.Join(", ", array.Items.Select(Expression))} }}",
            CompilerTupleExpression tuple => $"({string.Join(", ", tuple.Items.Select(Expression))})",
            CompilerIndexerExpression indexer => $"{Expression(indexer.Target)}[{Expression(indexer.Index)}]",
            CompilerInvocationExpression invocation => $"{Expression(invocation.Target)}({string.Join(", ", invocation.Arguments.Select(Expression))})",
            CompilerMemberAccessExpression member => $"{Expression(member.Target)}.{member.MemberName}",
            CompilerBinaryExpression binary => $"({Expression(binary.Left)} {binary.Operator} {Expression(binary.Right)})",
            CompilerUnaryExpression unary => $"({unary.Operator}{Expression(unary.Expression)})",
            _ => throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"C# expression '{expression.GetType().Name}' cannot be emitted.",
                stage: CompilerDiagnosticStage.Emit,
                span: expression.Span),
        };
    }

    private static string Literal(CompilerLiteral literal)
    {
        return literal switch
        {
            CompilerNullLiteral => "null",
            CompilerBooleanLiteral boolean => boolean.Value ? "true" : "false",
            CompilerSignedIntegerLiteral integer => integer.Value.ToString(CultureInfo.InvariantCulture),
            CompilerUnsignedIntegerLiteral integer => integer.Value.ToString(CultureInfo.InvariantCulture) + "UL",
            CompilerFloatingPointLiteral floating => floating.Value.ToString("R", CultureInfo.InvariantCulture)
                + (floating.ElementType == CompilerElementType.Float32 ? "f" : "d"),
            CompilerStringLiteral text => $"\"{Escape(text.Value)}\"",
            CompilerTensorLiteral tensor => TensorLiteral(tensor),
            _ => throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"Literal '{literal.GetType().Name}' cannot be emitted as C# syntax.",
                stage: CompilerDiagnosticStage.Emit,
                span: null),
        };
    }

    private static string TensorLiteral(CompilerTensorLiteral tensor)
    {
        var elementType = tensor.ElementType switch
        {
            CompilerElementType.Float32 => "float",
            CompilerElementType.Float64 => "double",
            CompilerElementType.Int8 => "sbyte",
            CompilerElementType.UInt8 => "byte",
            CompilerElementType.Int16 => "short",
            CompilerElementType.UInt16 => "ushort",
            CompilerElementType.Int32 => "int",
            CompilerElementType.UInt32 => "uint",
            CompilerElementType.Int64 => "long",
            CompilerElementType.UInt64 => "ulong",
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Scalar initializer dtype '{tensor.ElementType}' cannot be emitted as a TorchSharp tensor.",
                CompilerDiagnosticStage.Emit),
        };
        var values = string.Join(", ", tensor.Values.Select(Literal));
        var dimensions = string.Join(", ", tensor.Dimensions.Select(dimension =>
            dimension is CompilerFixedDimension fixedDimension
                ? fixedDimension.Value.ToString(CultureInfo.InvariantCulture) + "L"
                : throw new CSharpCompilerDiagnosticException(
                    CompilerDiagnosticCodes.Unsupported,
                    "Scalar initializer dimensions must be fixed.",
                    CompilerDiagnosticStage.Emit)));
        var torchType = tensor.ElementType switch
        {
            CompilerElementType.Float32 => "Float32",
            CompilerElementType.Float64 => "Float64",
            CompilerElementType.Int8 => "Int8",
            CompilerElementType.UInt8 => "UInt8",
            CompilerElementType.Int16 => "Int16",
            CompilerElementType.UInt16 => "UInt16",
            CompilerElementType.Int32 => "Int32",
            CompilerElementType.UInt32 => "UInt32",
            CompilerElementType.Int64 => "Int64",
            CompilerElementType.UInt64 => "UInt64",
            _ => throw new CSharpCompilerDiagnosticException(
                CompilerDiagnosticCodes.Unsupported,
                $"Scalar initializer dtype '{tensor.ElementType}' cannot be emitted as a TorchSharp tensor.",
                CompilerDiagnosticStage.Emit),
        };
        return $"torch.tensor(new {elementType}[] {{ {values} }}, [{dimensions}], dtype: torch.ScalarType.{torchType})";
    }

    private static bool AlwaysReturns(CompilerStatement statement)
    {
        return statement switch
        {
            CompilerReturnStatement => true,
            CompilerBlockStatement block => block.Statements.Any(AlwaysReturns),
            CompilerStaticIfStatement conditional => AlwaysReturns(conditional.WhenTrue)
                && conditional.WhenFalse is not null
                && AlwaysReturns(conditional.WhenFalse),
            _ => false,
        };
    }

    private void Line(string? value = null)
    {
        if (value is null)
        {
            _builder.AppendLine();
            return;
        }

        _builder.Append(' ', _indent * 4);
        _builder.AppendLine(value);
    }

    private static string Escape(string value) => value.Replace("\\", "\\\\").Replace("\"", "\\\"");
}
