package org.ml_methods_group.common.metrics.pickers;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.RepresentativePicker;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class ClusterRepresentativePicker implements RepresentativePicker<Solution> {

    private final ManyOptionsSelector<Solution, Solution> selector;
    private final DistanceFunction<Solution> metric;

    public ClusterRepresentativePicker(DistanceFunction<Solution> metric,
                                       ManyOptionsSelector<Solution, Solution> selector) {
        this.metric = metric;
        this.selector = selector;
    }

    @Override
    public Solution pick(List<Solution> incorrect) {
        final List<Solution> correct = incorrect.stream()
                .map(selector::selectOptions)
                .map(Optional::get)
                .flatMap(Collection::stream)
                .distinct()
                .collect(Collectors.toList());
        System.out.println(incorrect.size() + " " + correct.size());
        double minimalTotalDistanceToOthers = Double.MAX_VALUE;
        Solution center = null;
        for (Solution current : correct) {
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
