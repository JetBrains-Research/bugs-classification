package org.ml_methods_group.common.metrics.representatives;

import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.ManyRepresentativesPicker;
import org.ml_methods_group.common.Solution;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class HeuristicKMostFrequentPicker<V> implements ManyRepresentativesPicker<V> {

    private final ManyOptionsSelector<V, V> selector;
    private final int k;

    public HeuristicKMostFrequentPicker(ManyOptionsSelector<V, V> selector, int k) {
        this.selector = selector;
        this.k = k;
    }

    @Override
    public List<V> pick(List<V> incorrect) {
        final Map<V, Long> correctCounter = incorrect.stream()
                .map(selector::selectOptions)
                .map(Optional::get)
                .flatMap(Collection::stream)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        final var queue = new PriorityQueue<Map.Entry<V, Long>>(Map.Entry.comparingByValue());
        for (var entry : correctCounter.entrySet()) {
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
