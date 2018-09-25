package org.ml_methods_group.common;

import java.util.function.Function;

public class Wrapper<F, M> {
    private final F features;
    private final M meta;

    public Wrapper(F features, M meta) {
        this.features = features;
        this.meta = meta;
    }

    public F getFeatures() {
        return features;
    }

    public M getMeta() {
        return meta;
    }

    public static <T extends Comparable<? super T>, M extends Comparable<? super M>> int compare(Wrapper<T, M> first,
                                                                                                 Wrapper<T, M> second) {
        final int tmp = first.features.compareTo(second.features);
        return tmp != 0 ? tmp : first.meta.compareTo(second.meta);
    }

    public static <F, M> Function<M, Wrapper<F, M>> wrap(Function<M, F> processor) {
        return x -> new Wrapper<>(processor.apply(x), x);
    }

    @Override
    public String toString() {
        return "Wrapper{meta=" + meta + "}";
    }
}
