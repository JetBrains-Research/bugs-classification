package org.ml_methods_group.testing.metrics;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.changes.NodeType;
import org.ml_methods_group.common.changes.generation.ChangeGenerator;

import java.lang.ref.SoftReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class HeuristicChangesBasedDistanceFunction implements DistanceFunction<Solution> {
    private final Map<Integer, SoftReference<int[]>> counters = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> indexes = new ConcurrentHashMap<>();
    private final ChangeGenerator generator;
    private volatile int indexGenerator = 0;

    public HeuristicChangesBasedDistanceFunction(ChangeGenerator generator) {
        this.generator = generator;
    }

    @Override
    public double distance(Solution first, Solution second) {
        return generator.getChanges(first, second).getChanges().size();
    }

    @Override
    public double distance(Solution first, Solution second, double upperBound) {
        return checkHeuristic(first, second, upperBound) ? upperBound : Math.min(upperBound, distance(first, second));
    }

    private int[] getCountersFromCache(Solution solution) {
        final SoftReference<int[]> reference = counters.get(solution.getSolutionId());
        return reference == null ? null : reference.get();
    }

    private int[] getCounters(Solution solution) {
        int[] cached = getCountersFromCache(solution);
        if (cached != null) {
            return cached;
        }
        synchronized (indexes) {
            cached = getCountersFromCache(solution);
            if (cached != null) {
                return cached;
            }
            final int[] result = new int[NodeType.values().length];
            final ITree tree = generator.getTree(solution);
            tree.getTrees()
                    .stream()
                    .mapToInt(ITree::getType)
                    .map(type -> indexes.computeIfAbsent(type, x -> indexGenerator++))
                    .forEach(index -> result[index]++);
            counters.put(solution.getSolutionId(), new SoftReference<>(result));
            return result;
        }
    }

    private boolean checkHeuristic(Solution first, Solution second, double limit) {
        final int[] firstCounters = getCounters(first);
        final int[] secondCounters = getCounters(second);
        int total = 0;
        for (int i = 0; i < indexGenerator && total < limit; i++) {
            total += Math.abs(firstCounters[i] - secondCounters[i]);
        }
        return total >= limit;
    }
}
