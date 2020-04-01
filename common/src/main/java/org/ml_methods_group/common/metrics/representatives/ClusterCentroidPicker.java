package org.ml_methods_group.common.metrics.representatives;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.RepresentativePicker;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class ClusterCentroidPicker<V> implements RepresentativePicker<V> {

    private final ManyOptionsSelector<V, V> selector;
    private final DistanceFunction<V> metric;

    public ClusterCentroidPicker(DistanceFunction<V> metric,
                                 ManyOptionsSelector<V, V> selector) {
        this.metric = metric;
        this.selector = selector;
    }

    @Override
    public V pick(List<V> incorrect) {
        final List<V> correct = incorrect.stream()
                .map(selector::selectOptions)
                .map(Optional::get)
                .flatMap(Collection::stream)
                .distinct()
                .collect(Collectors.toList());
        System.out.println(incorrect.size() + " " + correct.size());
        double minimalTotalDistanceToOthers = Double.MAX_VALUE;
        V center = null;
        for (V current : correct) {
            double totalDistance = correct.stream()
                    .map(other -> metric.distance(current, other))
                    .mapToDouble(Double::doubleValue)
                    .sum();
            if (totalDistance < minimalTotalDistanceToOthers) {
                minimalTotalDistanceToOthers = totalDistance;
                center = current;
            }
        }
        return center;
    }
}
