package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.OptionSelector;

import java.util.*;
import java.util.stream.Collectors;

public class KClosestPairsSelector<V> implements OptionSelector<V, List<V>> {

    private final List<V> options;
    private final DistanceFunction<V> metric;
    private final int k;

    public KClosestPairsSelector(List<V> options, DistanceFunction<V> metric, int k_nearest) {
        this.options = new ArrayList<>(options);
        this.metric = metric;
        this.k = k_nearest;
    }

    class OptionsComparator implements Comparator<V> {

        private final V value;
        private final double minDistance;

        OptionsComparator(V value, double minDistance) {
            this.value = value;
            this.minDistance = minDistance;
        }

        @Override
        public int compare(V option1, V option2) {
            double dist1 = metric.distance(value, option1, minDistance);
            double dist2 = metric.distance(value, option2, minDistance);
            return Double.compare(dist1, dist2);
        }
    }

    @Override
    public Optional<List<V>> selectOption(V value) {
        if (options.isEmpty()) {
            return Optional.empty();
        }
        V closest = options.get(0);
        double minDistance = metric.distance(value, closest);

        return Optional.of(options.stream()
                .sorted(new OptionsComparator(value, minDistance))
                .limit(k)
                .collect(Collectors.toList()));
    }

    @Override
    public Collection<List<V>> getOptions() {
        return Collections.singleton(options);
    }
}
