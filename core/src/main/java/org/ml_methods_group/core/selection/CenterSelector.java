package org.ml_methods_group.core.selection;

import org.ml_methods_group.core.DistanceFunction;

import java.util.Comparator;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.Collectors;

public class CenterSelector<T> implements RepresenterSelector<T> {

    public enum Mode {
        MAX {
            @Override
            double getValue(DoubleSummaryStatistics statistics) {
                return statistics.getMax();
            }
        },
        AVERAGE {
            @Override
            double getValue(DoubleSummaryStatistics statistics) {
                return statistics.getAverage();
            }
        };
        abstract double getValue(DoubleSummaryStatistics statistics);
    }

    private final DistanceFunction<T> metric;
    private final double ignoring;
    private final Mode mode;

    public CenterSelector(DistanceFunction<T> metric, double ignoring, Mode mode) {
        this.metric = metric;
        this.ignoring = ignoring;
        this.mode = mode;
    }

    @Override
    public List<T> findRepresenter(int n, List<T> samples) {
        return samples.stream()
                .sorted(Comparator.comparingDouble(x -> mode.getValue(calculate(x, samples))))
                .limit(n)
                .collect(Collectors.toList());
    }

    private DoubleSummaryStatistics calculate(T object, List<T> samples) {
        return samples.stream()
                .mapToDouble(other -> metric.distance(object, other))
                .sorted()
                .limit((long) Math.ceil(samples.size() * (1 - ignoring)))
                .summaryStatistics();
    }
}
