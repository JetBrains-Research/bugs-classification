package org.ml_methods_group.core.basic.markers;

import org.ml_methods_group.core.Cluster;
import org.ml_methods_group.core.Marker;
import org.ml_methods_group.core.entities.Solution;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ExtrapolationMarker<V, M> implements Marker<Cluster<V>, M> {

    private final Marker<V, M> marker;
    private final int bound;

    public ExtrapolationMarker(Marker<V, M> marker, int bound) {
        this.marker = marker;
        this.bound = bound;
    }

    @Override
    public M mark(Cluster<V> cluster) {
        final List<M> tips = cluster.stream()
                .map(marker::mark)
                .filter(Objects::nonNull)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                .entrySet()
                .stream()
                .filter(entry -> entry.getValue() >= bound)
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());

        // todo remove
        System.out.println();
        System.out.println("Next Cluster");
        System.out.println(cluster.size() + " " + tips);
        List<V> l = cluster.elementsCopy();
        Collections.shuffle(l);
        l = l.subList(0, 5);
        l.forEach(x -> System.out.println(((Solution) x).getCode()));

//        return null;
        return tips.size() == 1 ? tips.get(0) : onFail(cluster);
    }

    protected M onFail(Cluster<V> cluster) {
        return null;
    }
}
