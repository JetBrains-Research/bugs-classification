package org.ml_methods_group.testing.extractors;

import org.ml_methods_group.common.FeaturesExtractor;

import java.util.HashMap;
import java.util.Map;
import java.util.function.ToIntFunction;

public class CachedFeaturesExtractor<V, F> implements FeaturesExtractor<V, F> {
    private final FeaturesExtractor<V, F> oracle;
    private final ToIntFunction<V> keyExtractor;
    private final Map<Integer, F> cache = new HashMap<>();

    public CachedFeaturesExtractor(FeaturesExtractor<V, F> oracle, ToIntFunction<V> keyExtractor) {
        this.oracle = oracle;
        this.keyExtractor = keyExtractor;
    }

    @Override
    public F process(V value) {
        final int key = keyExtractor.applyAsInt(value);
        return cache.computeIfAbsent(key, x -> oracle.process(value));
    }

    @Override
    public <R> FeaturesExtractor<V, R> compose(FeaturesExtractor<? super F, R> other) {
        throw new UnsupportedOperationException();
    }

    @Override
    public <R> FeaturesExtractor<R, F> extend(FeaturesExtractor<R, V> mapper) {
        throw new UnsupportedOperationException();
    }
}
