package org.ml_methods_group.common;

import java.io.Serializable;

@FunctionalInterface
public interface FeaturesExtractor<V, F> extends Serializable {
    F process(V value);

    default <R> FeaturesExtractor<V, R> compose(FeaturesExtractor<? super F, R> other) {
        return x -> other.process(process(x));
    }

    default <R> FeaturesExtractor<R, F> extend(FeaturesExtractor<R, V> mapper) {
        return x -> process(mapper.process(x));
    }
}
