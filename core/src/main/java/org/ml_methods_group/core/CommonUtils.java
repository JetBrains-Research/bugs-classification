package org.ml_methods_group.core;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class CommonUtils {
    public static <F, S, R> Function<F, R> compose(Function<F, S> first, Function<? super S, R> second) {
        return second.compose(first);
    }

    public static <F, S> Predicate<F> check(Function<F, S> function, Predicate<S> predicate) {
        return (Serializable & Predicate<F>) x -> predicate.test(function.apply(x));
    }

    public static <V, F> Predicate<V> checkEquals(Function<V, F> first, Function<V, F> second) {
        return (Serializable & Predicate<V>) x -> Objects.equals(first.apply(x), second.apply(x));
    }

    public static <K, N, V> Map<N, V> mapKey(Map<K, V> map, Function<K, N> remapping) {
        return map.entrySet()
                .stream()
                .collect(Collectors.toMap(remapping.compose(Map.Entry::getKey), Map.Entry::getValue));
    }

    public static <K, V1, V2, V3> Map<K, V3> combine(Map<K, V1> first, V1 firstDefault,
                                                     Map<K, V2> second, V2 secondDefault,
                                                     BiFunction<V1, V2, V3> combiner) {
        final HashMap<K, V3> result = new HashMap<>();
        for (Map.Entry<K, V1> entry : first.entrySet()) {
            final K key = entry.getKey();
            result.put(key, combiner.apply(entry.getValue(), second.getOrDefault(key, secondDefault)));
        }
        for (Map.Entry<K, V2> entry : second.entrySet()) {
            final K key = entry.getKey();
            result.computeIfAbsent(key, x -> combiner.apply(firstDefault, entry.getValue()));
        }
        return result;
    }
}
