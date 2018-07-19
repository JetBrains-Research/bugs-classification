package org.ml_methods_group.core.basic;

import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.core.DistanceFunction;
import org.ml_methods_group.core.changes.ChangeGenerator;
import org.ml_methods_group.core.entities.NodeType;
import org.ml_methods_group.core.entities.Solution;

import java.lang.ref.SoftReference;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.ml_methods_group.core.entities.NodeType.*;

public class HeuristicChangesBasedDistanceFunction implements DistanceFunction<Solution> {
    private final Map<Integer, SoftReference<int[]>> counters = new ConcurrentHashMap<>();
    private final ChangeGenerator generator;

    public HeuristicChangesBasedDistanceFunction(ChangeGenerator generator) {
        this.generator = generator;
    }

    @Override
    public double distance(Solution first, Solution second) {
        return generator.getChanges(first, second).size();
    }

    @Override
    public double distance(Solution first, Solution second, double upperBound) {
        return checkHeuristic(first, second, upperBound) ? upperBound : Math.min(upperBound, distance(first, second));
    }

    private int[] getCounters(Solution solution) {
        final SoftReference<int[]> reference = counters.get(solution.getSolutionId());
        if (reference != null && reference.get() != null) {
            return reference.get();
        }
        final int[] result = new int[NodeType.values().length];
        final ITree tree = generator.getTree(solution);
        tree.getTrees()
                .stream()
                .map(ITree::getType)
                .filter(type -> type != JAVADOC.ordinal() && type != BLOCK_COMMENT.ordinal() && type != LINE_COMMENT.ordinal())
                .forEach(type -> result[type]++);
        counters.put(solution.getSolutionId(), new SoftReference<int[]>(result));
        return result;
    }

    private boolean checkHeuristic(Solution first, Solution second, double limit) {
        final int[] firstCounters = getCounters(first);
        final int[] secondCounters = getCounters(second);
        int total = 0;
        for (int i = 0; i < firstCounters.length && total < limit; i++) {
            total += Math.abs(firstCounters[i] - secondCounters[i]);
        }
        return total >= limit;
    }
}
