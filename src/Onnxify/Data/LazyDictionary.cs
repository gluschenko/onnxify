using System.Collections.ObjectModel;

namespace Onnxify.Data;

internal class LazyDictionary<TKey, TValue> : KeyedCollection<TKey, TValue> where TKey : notnull
{
    private readonly Func<TValue, TKey> _keySelector;

    public LazyDictionary(Func<TValue, TKey> keySelector, IEqualityComparer<TKey>? comparer = null) : base(comparer)
    {
        _keySelector = keySelector;
    }

    protected override TKey GetKeyForItem(TValue item)
    {
        return _keySelector(item);
    }

    public new TValue this[TKey key]
    {
        get => base[key];
        set
        {
            var newKey = _keySelector(value);

            if (!Comparer.Equals(key, newKey))
            {
                throw new ArgumentException("Key of value does not match indexer key.");
            }

            if (TryGetValue(key, out var existing))
            {
                var index = Items.IndexOf(existing);
                SetItem(index, value);
            }
            else
            {
                Add(value);
            }
        }
    }

    public bool Replace(TKey key, TValue value)
    {
        var newKey = _keySelector(value);

        if (TryGetValue(newKey, out _) && !Comparer.Equals(key, newKey))
        {
            throw new InvalidOperationException($"Value with key '{newKey}' is already added.");
        }

        if (!TryGetValue(key, out var existing))
        {
            return false;
        }

        var index = Items.IndexOf(existing);
        SetItem(index, value);
        return true;
    }
}
