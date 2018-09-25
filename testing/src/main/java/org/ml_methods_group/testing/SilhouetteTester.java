package org.ml_methods_group.testing;

import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.Wrapper;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class SilhouetteTester<T, M> implements Tester<T, M> {
    private final DistanceFunction<Wrapper<T, M>> metric;

    public SilhouetteTester(DistanceFunction<Wrapper<T, M>> metric) {
        this.metric = metric;
    }

    @Override
    public SilhouetteResults test(List<List<Wrapper<T, M>>> clusters) {
        final double[] silhouettes = IntStream.range(0, clusters.size())
                .boxed()
                .flatMap(i -> clusters.get(i).stream()
                        .map(wrapper -> silhouette(wrapper, i, clusters)))
                .mapToDouble(Double::doubleValue)
                .toArray();
        final double silhouette = Arrays.stream(silhouettes)
                .average()
                .orElse(0);
        final int positiveSilhouettesCount = (int) Arrays.stream(silhouettes)
                .filter(x -> x > 0)
                .count();
        final int negativeSilhouettesCount = (int) Arrays.stream(silhouettes)
                .filter(x -> x < 0)
                .count();
        final int zeroSilhouettesCount = (int) Arrays.stream(silhouettes)
                .filter(x -> x == 0)
                .count();
        return new SilhouetteResults(silhouette, positiveSilhouettesCount,
                negativeSilhouettesCount, zeroSilhouettesCount);
    }

    private double silhouette(Wrapper<T, M> wrapper, List<Wrapper<T, M>> wrappers) {
        return wrappers.stream()
                .filter(other -> wrapper != other)
                .mapToDouble(other -> metric.distance(wrapper, other))
                .average()
                .orElse(0);
    }

    private double silhouette(Wrapper<T, M> wrapper, int clusterIndex, List<List<Wrapper<T, M>>> clusters) {
        if (clusters.get(clusterIndex).size() == 1) {
            return 0;
        }
        final double a = silhouette(wrapper, clusters.get(clusterIndex));
        final double b = IntStream.range(0, clusters.size())
                .filter(i -> i != clusterIndex)
                .mapToObj(clusters::get)
                .mapToDouble(cluster -> silhouette(wrapper, cluster))
                .min()
                .orElse(0);
        return (b - a) / Math.max(a, b);
    }

    public static class SilhouetteResults implements TestingResults {

        private final double silhouette;
        private final int positiveSilhouettesCount;
        private final int negativeSilhouettesCount;
        private final int zeroSilhouettesCount;

        private SilhouetteResults(double silhouette, int positiveSilhouettesCount,
                                  int negativeSilhouettesCount, int zeroSilhouettesCount) {
            this.silhouette = silhouette;
            this.positiveSilhouettesCount = positiveSilhouettesCount;
            this.negativeSilhouettesCount = negativeSilhouettesCount;
            this.zeroSilhouettesCount = zeroSilhouettesCount;
        }

        @Override
        public double getValue() {
            return (getSilhouette() + 1) / 2;
        }

        public double getSilhouette() {
            return silhouette;
        }

        public double getPositiveSilhouettesPart() {
            return (double) positiveSilhouettesCount /
                    (positiveSilhouettesCount + negativeSilhouettesCount + zeroSilhouettesCount);
        }

        public double getNegativeSilhouettesPart() {
            return (double) negativeSilhouettesCount /
                    (positiveSilhouettesCount + negativeSilhouettesCount + zeroSilhouettesCount);
        }

        @Override
        public String toString() {
            final int total = positiveSilhouettesCount + zeroSilhouettesCount + negativeSilhouettesCount;
            return "Testing results:\n" +
                    "\tPositive silhouettes:   " + positiveSilhouettesCount + "\t/\t" + total +
                    "\t=\t" + getPositiveSilhouettesPart() + "\n" +
                    "\tNegative silhouettes:   " + negativeSilhouettesCount + "\t/\t" + total +
                    "\t=\t" + getNegativeSilhouettesPart() + "\n" +
                    "\tSilhouette:             " + getSilhouette();
        }
    }
}
