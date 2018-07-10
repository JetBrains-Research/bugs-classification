package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.Solution;
import org.ml_methods_group.core.SolutionDiff;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class TestingUtils {
    public static List<Pair<Wrapper>> generatePairs(List<Wrapper> diffs, DistanceFunction<Wrapper> metric,
                                                    int count, Random random) {
        final List<Pair<Wrapper>> result = new ArrayList<>();
        final List<Wrapper> working = new ArrayList<>(diffs);
        for (int i = 0; i < count; i++) {
            final Wrapper first = diffs.get(random.nextInt(diffs.size()));
            final Wrapper second;
            if (random.nextBoolean()) {
                second = diffs.get(random.nextInt(diffs.size()));
            } else {
                working.sort(Comparator.comparingDouble(other -> metric.distance(first, other)));
                Wrapper another;
                do {
                    another = working.get(random.nextInt(Math.min(diffs.size(), 10)));
                } while (another == first);
                second = another;
            }
            result.add(new Pair<>(first, second));
        }
        return result;
    }
}
