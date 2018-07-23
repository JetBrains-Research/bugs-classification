package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.Wrapper;

import java.util.List;
import java.util.stream.IntStream;

public class InternalTester<T> implements Tester<T> {
    private final DistanceFunction<Wrapper<T>> metric;

    public InternalTester(DistanceFunction<Wrapper<T>> metric) {
        this.metric = metric;
    }

    @Override
    public double test(List<List<Wrapper<T>>> clusters) {
        return IntStream.range(0, clusters.size())
                .boxed()
                .flatMap(i -> clusters.get(i).stream()
                        .map(wrapper -> silhouette(wrapper, i, clusters)))
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);
    }

    private double silhouette(Wrapper<T> wrapper, List<Wrapper<T>> wrappers) {
        return wrappers.stream()
                .filter(other -> wrapper != other)
                .mapToDouble(other -> metric.distance(wrapper, other))
                .average()
                .orElse(2);
    }

    private double silhouette(Wrapper<T> wrapper, int clusterIndex, List<List<Wrapper<T>>> clusters) {
        final double a = silhouette(wrapper, clusters.get(clusterIndex));
        final double b = IntStream.range(0, clusters.size())
                .filter(i -> i != clusterIndex)
                .mapToObj(clusters::get)
                .mapToDouble(cluster -> silhouette(wrapper, cluster))
                .min()
                .orElse(2);
        return ((b - a) / (Math.max(a, b)) + 1) / 2;
    }
}
