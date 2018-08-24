package org.ml_methods_group.core.basic.selectors;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.OptionSelector;

import java.util.*;

public class ClosestPairSelector<V> implements OptionSelector<V, V> {

    private final List<V> options;
    private final DistanceFunction<V> metric;

    public ClosestPairSelector(List<V> options, DistanceFunction<V> metric) {
        this.options = new ArrayList<>(options);
        this.metric = metric;
    }

    @Override
    public Optional<V> selectOption(V value) {
        if (options.isEmpty()) {
            return Optional.empty();
        }
        V closest = options.get(0);
        double minDistance = metric.distance(value, closest);
        for (V option : options.subList(1, options.size())) {
            final double distance = metric.distance(value, option, minDistance);
            if (distance < minDistance) {
                minDistance = distance;
                closest = option;
            }
        }
        return Optional.of(closest);
    }

    @Override
    public List<V> getOptions() {
        return Collections.unmodifiableList(options);
    }
}
