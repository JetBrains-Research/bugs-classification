package org.ml_methods_group.common.metrics.representatives;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.RepresentativesPicker;

import java.util.Collections;
import java.util.List;

public class CentroidPicker<V> implements RepresentativesPicker<V> {

    private final DistanceFunction<V> metric;

    public CentroidPicker(DistanceFunction<V> metric) {
        this.metric = metric;
    }

    @Override
    public List<V> getRepresentatives(Cluster<V> values) {
        if (values.getElements().isEmpty()) {
            return List.of();
        }
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
        return Collections.singletonList(center);
    }

}
