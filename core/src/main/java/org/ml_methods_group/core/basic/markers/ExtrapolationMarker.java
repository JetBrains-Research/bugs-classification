package org.ml_methods_group.core.basic.markers;

import org.ml_methods_group.core.ClusterMarker;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ExtrapolationMarker<V, K, M> implements ClusterMarker<V, M> {

    private final Map<K, M> marks;
    private final Function<V, K> keyExtractor;
    private final int bound;
    private final ClusterMarker<V, M> oracle;

    public ExtrapolationMarker(Map<V, M> marks, Function<V, K> keyExtractor, int bound, ClusterMarker<V, M> oracle) {
        this.oracle = oracle;
        this.bound = bound;
        this.marks = marks.entrySet()
                .stream()
                .collect(Collectors.toMap(keyExtractor.compose(Map.Entry::getKey), Map.Entry::getValue));
        this.keyExtractor = keyExtractor;
    }
    
    @Override
    public M mark(List<V> cluster) {
        final List<M> tips = cluster.stream()
                .map(keyExtractor)
                .map(marks::get)
                .filter(Objects::nonNull)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                .entrySet()
                .stream()
                .filter(entry -> entry.getValue() >= bound)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
        return tips.size() == 1 ? tips.get(0) : oracle.mark(cluster);
    }
}
