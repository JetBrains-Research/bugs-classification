package org.ml_methods_group.common.metrics.representatives;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.RepresentativesProducer;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class KMostFrequentOptionsProducer<V, O> implements RepresentativesProducer<V, O> {

    private final ManyOptionsSelector<V, O> selector;
    private final int k;

    public KMostFrequentOptionsProducer(ManyOptionsSelector<V, O> selector, int k) {
        this.selector = selector;
        this.k = k;
    }

    @Override
    public List<O> getRepresentatives(Cluster<V> values) {
        final Map<O, Long> optionsCounter = values.stream()
                .map(selector::selectOptions)
                .map(Optional::get)
                .flatMap(Collection::stream)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        final var queue = new PriorityQueue<Map.Entry<O, Long>>(Map.Entry.comparingByValue());
        for (var entry : optionsCounter.entrySet()) {
            queue.offer(entry);
            if (queue.size() > k) {
                queue.poll();
            }
        }
        return queue.stream()
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    @Override
    public int getSelectionSize() {
        return k;
    }
}
