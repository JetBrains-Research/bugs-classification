package org.ml_methods_group.common.metrics.pickers;

import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.ManyRepresentativesPicker;
import org.ml_methods_group.common.Solution;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class HeuristicKRepresentativesPicker implements ManyRepresentativesPicker<Solution> {

    private final ManyOptionsSelector<Solution, Solution> selector;
    private final int k;

    public HeuristicKRepresentativesPicker(ManyOptionsSelector<Solution, Solution> selector, int k) {
        this.selector = selector;
        this.k = k;
    }

    @Override
    public List<Solution> pick(List<Solution> incorrect) {
        final Map<Solution, Long> correctCounter = incorrect.stream()
                .map(selector::selectOptions)
                .map(Optional::get)
                .flatMap(Collection::stream)
                .collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        final var queue = new PriorityQueue<Map.Entry<Solution, Long>>(Map.Entry.comparingByValue());
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
