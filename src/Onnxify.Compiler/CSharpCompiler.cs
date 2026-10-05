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

/// <summary>Сеанс импорта и генерации C# TorchSharp через общее compiler IR.</summary>
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
    /// Разбирает заданный исходник или декомпилирует указанный метод и преобразует его в IR.
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
    /// Декомпилирует только выбранный metadata token и передаёт его текст тому же scanner, что используется для source text.
    /// Это сохраняет единый путь анализа и не позволяет decompiler AST проникнуть в публичные compiler contracts.
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

        var method = FindMethod(moduleType, descriptor);
        var settings = new DecompilerSettings(ICSharpCode.Decompiler.CSharp.LanguageVersion.CSharp10_0)
        {
            ThrowOnAssemblyResolveErrors = false,
        };
        var decompiler = new CSharpDecompiler(descriptor.AssemblyPath, settings);
        var syntaxTree = decompiler.Decompile(
            MetadataTokenHelpers.EntityHandleOrNil(method.MetadataToken));
        var decompiledMethod = syntaxTree
            .Descendants
            .OfType<ICSharpCode.Decompiler.CSharp.Syntax.MethodDeclaration>()
            .FirstOrDefault(x => string.Equals(x.Name, method.Name, StringComparison.Ordinal));

        if (decompiledMethod is null)
        {
            throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.InvalidSource,
                message: $"Method '{descriptor.MethodName}' could not be decompiled.",
                stage: CompilerDiagnosticStage.Parse,
                span: Span(descriptor.Document, 0, 0));
        }

        // The decompiler is intentionally limited to the requested token. Parsing its textual
        // method representation gives the scanner a stable compiler-owned boundary and prevents
        // raw decompiler nodes from escaping into the public API.
        return ImportSourceText(
            decompiledMethod.ToString(),
            descriptor.Document,
            new CompilationContext(descriptor, method));
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
            CompilerTorchSharpModuleDescriptor descriptor,
            MethodInfo method)
        {
            Descriptor = descriptor;
            Method = method;
        }

        public CompilerTorchSharpModuleDescriptor Descriptor { get; }

        public MethodInfo Method { get; }
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
    /// Сканирует выбранный method body и переносит контракты, state и helper методы в compiler IR.
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
        var builder = CreateBuilder(method, descriptor);
        RegisterValueContracts(builder, inputs, outputs);
        RegisterStateMetadata(builder, descriptor);

        var body = ScanBody(method);
        AddDeclaredValues(
            builder: builder,
            body: body,
            inputs: inputs,
            outputs: outputs,
            descriptor: descriptor);
        AddHelperBlocks(builder, helperMethods);

        builder.SetSyntaxBody(body);

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

    private void AddHelperBlocks(
        CompilerComputationTreeBuilder builder,
        IEnumerable<MethodDeclarationSyntax>? helperMethods
    )
    {
        foreach (var helper in helperMethods ?? Array.Empty<MethodDeclarationSyntax>())
        {
            var helperInputs = GetInputs(helper, descriptor: null);
            var helperOutputs = GetOutputs(helper, descriptor: null);
            var block = new CompilerComputationBlock(
                name: helper.Identifier.ValueText,
                inputs: helperInputs.Select(x => new CompilerValueReference(x.Name)),
                outputs: helperOutputs.Select(x => new CompilerValueReference(x.Name)),
                body: ScanBody(helper),
                span: Span(helper));
            builder.AddBlock(block);
        }
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
        return expression.ToString().Contains("no_grad", StringComparison.Ordinal);
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
                            new CompilerIndexerExpression(
                                ScanExpression(variable.Initializer.Value),
                                new CompilerLiteralExpression(
                                    new CompilerSignedIntegerLiteral(
                                        CompilerElementType.Int32,
                                        index)),
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
    /// Преобразует поддержанные C# expressions в compiler-owned syntax IR с исходными spans.
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
                    invocation.ArgumentList.Arguments.Select(x => ScanExpression(x.Expression)),
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
                caller: _context?.Descriptor.MethodName ?? "forward",
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
    /// Генерирует исходный TorchSharp-класс из syntax IR и возвращает diagnostics для конструкций без C# mapping.
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
            if (tree.Operations.OfType<CompilerOperation>().Any())
            {
                return CompilerResult<string>.Failure(
                [
                    new CompilerDiagnostic(
                        code: CompilerDiagnosticCodes.Unsupported,
                        message: "ONNX operations do not have C# TorchSharp mappings in OXY-023; operator mappings are implemented by OXY-024.",
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
    /// Собирает C# модуль из compiler-owned IR, сохраняя порядок блоков и инструкций.
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
            if (operation is not CompilerModuleCall call)
            {
                throw new CSharpCompilerDiagnosticException(
                    code: CompilerDiagnosticCodes.Unsupported,
                    message: $"Operation '{operation.Name}' cannot be emitted as C# TorchSharp code.",
                    stage: CompilerDiagnosticStage.Emit,
                    span: operation.Span);
            }

            var inputs = string.Join(", ", call.Inputs.Where(x => !x.IsEmptyOptional).Select(x => x.Name));
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
        var outputType = block.Outputs.Count == 1
            ? TORCH_TENSOR_TYPE
            : $"({string.Join(", ", block.Outputs.Select(x => $"{TORCH_TENSOR_TYPE} {x.Name}"))})";
        Line($"private {outputType} {block.Name}({string.Join(", ", block.Inputs.Select(x => $"{TORCH_TENSOR_TYPE} {x.Name}"))})");
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
            CompilerFloatingPointLiteral floating => floating.Value.ToString("R", CultureInfo.InvariantCulture) + "d",
            CompilerStringLiteral text => $"\"{Escape(text.Value)}\"",
            _ => throw new CSharpCompilerDiagnosticException(
                code: CompilerDiagnosticCodes.Unsupported,
                message: $"Literal '{literal.GetType().Name}' cannot be emitted as C# syntax.",
                stage: CompilerDiagnosticStage.Emit,
                span: null),
        };
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
