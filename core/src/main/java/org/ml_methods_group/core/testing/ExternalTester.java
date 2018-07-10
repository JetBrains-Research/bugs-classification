package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.changes.ChangeType;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.ml_methods_group.core.testing.ExternalTester.PairGuess.NEUTRAL;
import static org.ml_methods_group.core.testing.ExternalTester.PairGuess.SIMILAR;

public class ExternalTester implements Tester {
    private final Index<Pair<Integer>, PairGuess> index;

    public ExternalTester(Index<Pair<Integer>, PairGuess> index) {
        this.index = index;
    }

    @Override
    public double test(List<List<Wrapper>> clusters) {
        final HashMap<Integer, Integer> sessionToCluster = new HashMap<>();
        for (int i = 0; i < clusters.size(); i++) {
            final int clusterIndex = i;
            clusters.get(i).stream()
                    .map(wrapper -> wrapper.sessionId)
                    .forEach(id -> sessionToCluster.put(id, clusterIndex));
        }
        int errors = 0;
        int total = 0;
        for (Map.Entry<Pair<Integer>, PairGuess> test : index.getIndex().entrySet()) {
            final int first = sessionToCluster.getOrDefault(test.getKey().first, -1);
            final int second = sessionToCluster.getOrDefault(test.getKey().second, -1);
            if (test.getValue() != NEUTRAL && first != -1 && second != -1) {
                total++;
                errors += (first == second) ^ test.getValue() == SIMILAR ? 1 : 0;
            }
        }
        return 1 - (double) errors / total;
    }

    public enum PairGuess {
        SIMILAR, NEUTRAL, DIFFERENT;

        private static final PairGuess[] buffer = values();

        public static PairGuess valueOf(int value) {
            return value == -1 ? null : buffer[value];
        }
    }
}
