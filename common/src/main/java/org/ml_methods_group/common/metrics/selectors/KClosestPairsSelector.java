package org.ml_methods_group.common.metrics.selectors;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.ManyOptionsSelector;

import java.util.*;

public class KClosestPairsSelector<V> implements ManyOptionsSelector<V, V> {

    private final List<V> options;
    private final DistanceFunction<V> metric;
    private final int k;

    public KClosestPairsSelector(List<V> options, DistanceFunction<V> metric, int k_nearest) {
        this.options = options;
        this.metric = metric;
        this.k = k_nearest;
    }

    @Override
    public Optional<List<V>> selectOptions(V value) {
        if (options.isEmpty()) {
            return Optional.empty();
        }
        List<V> kClosest = new ArrayList<>(Collections.nCopies(k, options.get(0)));
        List<Double> kClosestDistances = new ArrayList<>(Collections.nCopies(k, Double.MAX_VALUE));
        for (V option : options) {
            final double currentDistance = metric.distance(value, option, Double.MAX_VALUE);
            for (int i = 0; i < k; ++i) {
                double kthDistance = kClosestDistances.get(i);
                if (kthDistance > currentDistance) {
                    for (int j = k - 1; j > i; j--) {
                        kClosest.set(j, kClosest.get(j - 1));
                        kClosestDistances.set(j, kClosestDistances.get(j - 1));
                    }
                    kClosestDistances.set(i, currentDistance);
                    kClosest.set(i, option);
                    break;
                }
            }
        }
        for (var dist : kClosestDistances)
            System.out.print(dist + " ");
        System.out.println();
        return Optional.of(kClosest);
    }

    @Override
    public Collection<V> getAllPossibleOptions() {
        return options;
    }

    @Override
    public int getKNearest() { return k; }
}
