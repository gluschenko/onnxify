namespace Onnxify.Compiler;

/// <summary>Base class for compiler-owned C# expressions.</summary>
public abstract class CompilerExpression : IEquatable<CompilerExpression>
{
    protected CompilerExpression(CompilerSourceSpan? span = null)
    {
        Span = span;
    }

    public CompilerSourceSpan? Span { get; }

    public bool Equals(CompilerExpression? other)
    {
        return other is not null
            && GetType() == other.GetType()
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span)
            && EqualsCore(other);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerExpression);

    public override int GetHashCode() => CompilerStructural.Combine(17, Span, GetHashCodeCore());

    protected abstract bool EqualsCore(CompilerExpression other);

    protected abstract int GetHashCodeCore();
}

public sealed class CompilerReferenceExpression : CompilerExpression
{
    public CompilerReferenceExpression(string name, CompilerSourceSpan? span = null)
        : base(span)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Reference name cannot be empty.", nameof(name));
        }

        Name = name;
    }

    public string Name { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        return string.Equals(Name, ((CompilerReferenceExpression)other).Name, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => Name.GetHashCode();
}

public sealed class CompilerLiteralExpression : CompilerExpression
{
    public CompilerLiteralExpression(CompilerLiteral literal, CompilerSourceSpan? span = null)
        : base(span)
    {
        CompilerStructural.RequireNotNull(literal, nameof(literal));
        Literal = literal;
    }

    public CompilerLiteral Literal { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        return EqualityComparer<CompilerLiteral>.Default.Equals(Literal, ((CompilerLiteralExpression)other).Literal);
    }

    protected override int GetHashCodeCore() => Literal.GetHashCode();
}

public sealed class CompilerArrayExpression : CompilerExpression
{
    public CompilerArrayExpression(IEnumerable<CompilerExpression> items, CompilerSourceSpan? span = null)
        : base(span)
    {
        Items = CompilerStructural.Copy(items, nameof(items));
    }

    public IReadOnlyList<CompilerExpression> Items { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        return CompilerStructural.SequenceEqual(Items, ((CompilerArrayExpression)other).Items);
    }

    protected override int GetHashCodeCore() => CompilerStructural.GetHashCode(Items);
}

public sealed class CompilerTupleExpression : CompilerExpression
{
    public CompilerTupleExpression(IEnumerable<CompilerExpression> items, CompilerSourceSpan? span = null)
        : base(span)
    {
        Items = CompilerStructural.Copy(items, nameof(items));
    }

    public IReadOnlyList<CompilerExpression> Items { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        return CompilerStructural.SequenceEqual(Items, ((CompilerTupleExpression)other).Items);
    }

    protected override int GetHashCodeCore() => CompilerStructural.GetHashCode(Items);
}

public sealed class CompilerIndexerExpression : CompilerExpression
{
    public CompilerIndexerExpression(
        CompilerExpression target,
        CompilerExpression index,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(target, nameof(target));
        CompilerStructural.RequireNotNull(index, nameof(index));
        Target = target;
        Index = index;
    }

    public CompilerExpression Target { get; }

    public CompilerExpression Index { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        var indexer = (CompilerIndexerExpression)other;
        return EqualityComparer<CompilerExpression>.Default.Equals(Target, indexer.Target)
            && EqualityComparer<CompilerExpression>.Default.Equals(Index, indexer.Index);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Target, Index);
}

public sealed class CompilerInvocationExpression : CompilerExpression
{
    public CompilerInvocationExpression(
        CompilerExpression target,
        IEnumerable<CompilerExpression> arguments,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(target, nameof(target));
        Target = target;
        Arguments = CompilerStructural.Copy(arguments, nameof(arguments));
    }

    public CompilerExpression Target { get; }

    public IReadOnlyList<CompilerExpression> Arguments { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        var invocation = (CompilerInvocationExpression)other;
        return EqualityComparer<CompilerExpression>.Default.Equals(Target, invocation.Target)
            && CompilerStructural.SequenceEqual(Arguments, invocation.Arguments);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Target, CompilerStructural.GetHashCode(Arguments));
}

/// <summary>Represents member access such as <c>module.forward</c> or <c>torch.Tensor</c>.</summary>
public sealed class CompilerMemberAccessExpression : CompilerExpression
{
    public CompilerMemberAccessExpression(
        CompilerExpression target,
        string memberName,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(target, nameof(target));
        if (string.IsNullOrWhiteSpace(memberName))
        {
            throw new ArgumentException("Member name cannot be empty.", nameof(memberName));
        }

        Target = target;
        MemberName = memberName;
    }

    public CompilerExpression Target { get; }

    public string MemberName { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        var member = (CompilerMemberAccessExpression)other;
        return EqualityComparer<CompilerExpression>.Default.Equals(Target, member.Target)
            && string.Equals(MemberName, member.MemberName, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Target, MemberName);
}

/// <summary>Represents a C# binary operator while retaining its source spelling.</summary>
public sealed class CompilerBinaryExpression : CompilerExpression
{
    public CompilerBinaryExpression(
        CompilerExpression left,
        string @operator,
        CompilerExpression right,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(left, nameof(left));
        CompilerStructural.RequireNotNull(right, nameof(right));
        if (string.IsNullOrWhiteSpace(@operator))
        {
            throw new ArgumentException("Binary operator cannot be empty.", nameof(@operator));
        }

        Left = left;
        Operator = @operator;
        Right = right;
    }

    public CompilerExpression Left { get; }

    public string Operator { get; }

    public CompilerExpression Right { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        var binary = (CompilerBinaryExpression)other;
        return EqualityComparer<CompilerExpression>.Default.Equals(Left, binary.Left)
            && string.Equals(Operator, binary.Operator, StringComparison.Ordinal)
            && EqualityComparer<CompilerExpression>.Default.Equals(Right, binary.Right);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Left, Operator, Right);
}

/// <summary>Represents a C# unary operator while retaining its source spelling.</summary>
public sealed class CompilerUnaryExpression : CompilerExpression
{
    public CompilerUnaryExpression(
        string @operator,
        CompilerExpression expression,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(expression, nameof(expression));
        if (string.IsNullOrWhiteSpace(@operator))
        {
            throw new ArgumentException("Unary operator cannot be empty.", nameof(@operator));
        }

        Operator = @operator;
        Expression = expression;
    }

    public string Operator { get; }

    public CompilerExpression Expression { get; }

    protected override bool EqualsCore(CompilerExpression other)
    {
        var unary = (CompilerUnaryExpression)other;
        return string.Equals(Operator, unary.Operator, StringComparison.Ordinal)
            && EqualityComparer<CompilerExpression>.Default.Equals(Expression, unary.Expression);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Operator, Expression);
}

/// <summary>Base class for ordered compiler-owned C# statements.</summary>
public abstract class CompilerStatement : IEquatable<CompilerStatement>
{
    protected CompilerStatement(CompilerSourceSpan? span = null)
    {
        Span = span;
    }

    public CompilerSourceSpan? Span { get; }

    public bool Equals(CompilerStatement? other)
    {
        return other is not null
            && GetType() == other.GetType()
            && EqualityComparer<CompilerSourceSpan?>.Default.Equals(Span, other.Span)
            && EqualsCore(other);
    }

    public override bool Equals(object? obj) => Equals(obj as CompilerStatement);

    public override int GetHashCode() => CompilerStructural.Combine(17, Span, GetHashCodeCore());

    protected abstract bool EqualsCore(CompilerStatement other);

    protected abstract int GetHashCodeCore();
}

public sealed class CompilerBlockStatement : CompilerStatement
{
    public CompilerBlockStatement(IEnumerable<CompilerStatement> statements, CompilerSourceSpan? span = null)
        : base(span)
    {
        Statements = CompilerStructural.Copy(statements, nameof(statements));
    }

    public IReadOnlyList<CompilerStatement> Statements { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        return CompilerStructural.SequenceEqual(Statements, ((CompilerBlockStatement)other).Statements);
    }

    protected override int GetHashCodeCore() => CompilerStructural.GetHashCode(Statements);
}

public sealed class CompilerDeclarationStatement : CompilerStatement
{
    public CompilerDeclarationStatement(
        string name,
        CompilerExpression? initializer = null,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Declaration name cannot be empty.", nameof(name));
        }

        Name = name;
        Initializer = initializer;
    }

    public string Name { get; }

    public CompilerExpression? Initializer { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        var declaration = (CompilerDeclarationStatement)other;
        return string.Equals(Name, declaration.Name, StringComparison.Ordinal)
            && EqualityComparer<CompilerExpression?>.Default.Equals(Initializer, declaration.Initializer);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Name, Initializer);
}

public sealed class CompilerAssignmentStatement : CompilerStatement
{
    public CompilerAssignmentStatement(
        CompilerExpression target,
        CompilerExpression value,
        CompilerSourceSpan? span = null,
        string @operator = "="
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(target, nameof(target));
        CompilerStructural.RequireNotNull(value, nameof(value));
        if (string.IsNullOrWhiteSpace(@operator))
        {
            throw new ArgumentException("Assignment operator cannot be empty.", nameof(@operator));
        }

        Target = target;
        Value = value;
        Operator = @operator;
    }

    public CompilerExpression Target { get; }

    public CompilerExpression Value { get; }

    public string Operator { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        var assignment = (CompilerAssignmentStatement)other;
        return EqualityComparer<CompilerExpression>.Default.Equals(Target, assignment.Target)
            && EqualityComparer<CompilerExpression>.Default.Equals(Value, assignment.Value)
            && string.Equals(Operator, assignment.Operator, StringComparison.Ordinal);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Target, Value, Operator);
}

public sealed class CompilerReturnStatement : CompilerStatement
{
    public CompilerReturnStatement(CompilerExpression? expression = null, CompilerSourceSpan? span = null)
        : base(span)
    {
        Expression = expression;
    }

    public CompilerExpression? Expression { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        return EqualityComparer<CompilerExpression?>.Default.Equals(Expression, ((CompilerReturnStatement)other).Expression);
    }

    protected override int GetHashCodeCore() => Expression?.GetHashCode() ?? 0;
}

public sealed class CompilerExpressionStatement : CompilerStatement
{
    public CompilerExpressionStatement(CompilerExpression expression, CompilerSourceSpan? span = null)
        : base(span)
    {
        CompilerStructural.RequireNotNull(expression, nameof(expression));
        Expression = expression;
    }

    public CompilerExpression Expression { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        return EqualityComparer<CompilerExpression>.Default.Equals(Expression, ((CompilerExpressionStatement)other).Expression);
    }

    protected override int GetHashCodeCore() => Expression.GetHashCode();
}

public sealed class CompilerStaticIfStatement : CompilerStatement
{
    public CompilerStaticIfStatement(
        CompilerExpression condition,
        CompilerStatement whenTrue,
        CompilerStatement? whenFalse = null,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        CompilerStructural.RequireNotNull(condition, nameof(condition));
        CompilerStructural.RequireNotNull(whenTrue, nameof(whenTrue));
        Condition = condition;
        WhenTrue = whenTrue;
        WhenFalse = whenFalse;
    }

    public CompilerExpression Condition { get; }

    public CompilerStatement WhenTrue { get; }

    public CompilerStatement? WhenFalse { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        var conditional = (CompilerStaticIfStatement)other;
        return EqualityComparer<CompilerExpression>.Default.Equals(Condition, conditional.Condition)
            && EqualityComparer<CompilerStatement>.Default.Equals(WhenTrue, conditional.WhenTrue)
            && EqualityComparer<CompilerStatement?>.Default.Equals(WhenFalse, conditional.WhenFalse);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, Condition, WhenTrue, WhenFalse);
}

public sealed class CompilerStaticForeachStatement : CompilerStatement
{
    public CompilerStaticForeachStatement(
        string variableName,
        CompilerExpression collection,
        CompilerStatement body,
        CompilerSourceSpan? span = null
    ) : base(span)
    {
        if (string.IsNullOrWhiteSpace(variableName))
        {
            throw new ArgumentException("Foreach variable name cannot be empty.", nameof(variableName));
        }

        CompilerStructural.RequireNotNull(collection, nameof(collection));
        CompilerStructural.RequireNotNull(body, nameof(body));
        VariableName = variableName;
        Collection = collection;
        Body = body;
    }

    public string VariableName { get; }

    public CompilerExpression Collection { get; }

    public CompilerStatement Body { get; }

    protected override bool EqualsCore(CompilerStatement other)
    {
        var loop = (CompilerStaticForeachStatement)other;
        return string.Equals(VariableName, loop.VariableName, StringComparison.Ordinal)
            && EqualityComparer<CompilerExpression>.Default.Equals(Collection, loop.Collection)
            && EqualityComparer<CompilerStatement>.Default.Equals(Body, loop.Body);
    }

    protected override int GetHashCodeCore() => CompilerStructural.Combine(17, VariableName, Collection, Body);
}
