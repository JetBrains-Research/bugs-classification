package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.ManyOptionsSelector;

import java.util.*;

public class KClosestPairsSelector<V> implements ManyOptionsSelector<V, V> {

    private final List<V> options;
    private final DistanceFunction<V> metric;
    private final int k;

    public KClosestPairsSelector(List<V> options, DistanceFunction<V> metric, int kNearest) {
        this.options = options;
        this.metric = metric;
        this.k = kNearest;
    }

    @Override
    public Optional<List<V>> selectOptions(V value) {
        if (options.isEmpty()) {
            return Optional.empty();
        }
        List<V> kClosest = new ArrayList<>(Collections.nCopies(k, options.get(0)));
        List<Double> kClosestDistances = new ArrayList<>(Collections.nCopies(k, Double.MAX_VALUE));
        for (V option : options) {
            final double currentDistance = metric.distance(value, option, kClosestDistances.get(k - 1));
            for (int i = 0; i < k; ++i) {
                double kthDistance = kClosestDistances.get(i);
                if (kClosestDistances.get(i) > currentDistance) {
                    kClosest.add(i, option);
                    kClosest.remove(k);
                    kClosestDistances.add(i, currentDistance);
                    kClosestDistances.remove(k);
                    break;
                }
            }
        }
        return Optional.of(kClosest);
    }

    @Override
    public Collection<V> getAllPossibleOptions() {
        return options;
    }

    @Override
    public int getSelectionSize() { return k; }
}
