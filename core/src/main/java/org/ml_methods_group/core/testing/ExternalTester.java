package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.TestPair;
import org.ml_methods_group.core.vectorization.Wrapper;

import java.util.HashMap;
import java.util.List;

import static org.ml_methods_group.core.entities.PairGuess.NEUTRAL;
import static org.ml_methods_group.core.entities.PairGuess.SIMILAR;

public class ExternalTester implements Tester {
    private final Repository<TestPair> tests;

    public ExternalTester(Repository<TestPair> tests) {
        this.tests = tests;
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
        for (TestPair test : tests) {
            final int first = sessionToCluster.getOrDefault(test.getFirstSessionId(), -1);
            final int second = sessionToCluster.getOrDefault(test.getSecondSessionId(), -1);
            if (test.getGuess() != NEUTRAL && first != -1 && second != -1) {
                total++;
                //todo
                errors += (first == second) ^ test.getGuess() == SIMILAR ? 1 : 0;
            }
        }
        return 1 - (double) errors / total;
    }
}
