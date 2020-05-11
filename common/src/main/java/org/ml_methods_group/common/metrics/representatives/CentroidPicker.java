package org.ml_methods_group.common.metrics.representatives;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.ManyOptionsSelector;
import org.ml_methods_group.common.RepresentativesPicker;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class CentroidPicker<V> implements RepresentativesPicker<V> {

    private final DistanceFunction<V> metric;

    public CentroidPicker(DistanceFunction<V> metric) {
        this.metric = metric;
    }

    @Override
    public List<V> getRepresentatives(Cluster<V> values) {
        double minimalTotalDistanceToOthers = Double.MAX_VALUE;
        V center = values.getElements().get(0);
        for (V current : values) {
            double totalDistance = values.stream()
                    .map(other -> metric.distance(current, other))
                    .mapToDouble(Double::doubleValue)
                    .sum();
            if (totalDistance < minimalTotalDistanceToOthers) {
                minimalTotalDistanceToOthers = totalDistance;
                center = current;
            }
        }
        return List.of(center);
    }

}
