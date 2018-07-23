package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.Wrapper;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.TestPair;

import java.util.HashMap;
import java.util.List;

import static org.ml_methods_group.core.entities.PairGuess.DIFFERENT;
import static org.ml_methods_group.core.entities.PairGuess.SIMILAR;

public class PairsTester<T> implements Tester<T> {
    private final Repository<TestPair> tests;

    public PairsTester(Repository<TestPair> tests) {
        this.tests = tests;
    }

    @Override
    public PairsTestingResult test(List<List<Wrapper<T>>> clusters) {
        final HashMap<Integer, Integer> sessionToCluster = new HashMap<>();
        for (int i = 0; i < clusters.size(); i++) {
            final int clusterIndex = i;
            clusters.get(i).stream()
                    .map(Wrapper::getSessionId)
                    .forEach(id -> sessionToCluster.put(id, clusterIndex));
        }
        int similarPairsCount = 0;
        int similarPairsErrors = 0;
        int differentPairsCount = 0;
        int differentPairsErrors = 0;
        for (TestPair test : tests) {
            final int first = sessionToCluster.getOrDefault(test.getFirstSessionId(), -1);
            final int second = sessionToCluster.getOrDefault(test.getSecondSessionId(), -1);
            if (first == -1 || second == -1) {
                continue;
            }
            if (test.getGuess() == SIMILAR) {
                similarPairsCount++;
                similarPairsErrors += (first != second ? 1 : 0);
            } else if (test.getGuess() == DIFFERENT) {
                differentPairsCount++;
                differentPairsErrors += (first == second ? 1 : 0);
            }
        }
        return new PairsTestingResult(similarPairsCount, similarPairsErrors, differentPairsCount, differentPairsErrors);
    }

    public static class PairsTestingResult implements TestingResults {

        private final int similarPairs;
        private final int similarPairsErrors;
        private final int differentPairs;
        private final int differentPairsErrors;

        public PairsTestingResult(int similarPairs, int similarPairsErrors,
                                  int differentPairs, int differentPairsErrors) {
            this.similarPairs = similarPairs;
            this.similarPairsErrors = similarPairsErrors;
            this.differentPairs = differentPairs;
            this.differentPairsErrors = differentPairsErrors;
        }

        @Override
        public double getValue() {
            return getAccuracy();
        }

        public double getAccuracy() {
            return 1 - (double) (similarPairsErrors + differentPairsErrors) / (similarPairs + differentPairs);
        }

        public double getSimilarPairsAccuracy() {
            return 1 - (double) similarPairsErrors / similarPairs;
        }

        public double getDifferentPairsAccuracy() {
            return 1 - (double) differentPairsErrors / differentPairs;
        }

        @Override
        public String toString() {
            return "Testing results:\n" +
                    "\tSimilar pairs accuracy:   " + (similarPairs - similarPairsErrors) + "\t/\t" + similarPairs +
                    "\t=\t" + getSimilarPairsAccuracy() + "\n" +
                    "\tDifferent pairs accuracy: " + (differentPairs - differentPairsErrors) + "\t/\t" + differentPairs +
                    "\t=\t" + getDifferentPairsAccuracy() + "\n" +
                    "\tTotal accuracy:           " +
                    (similarPairs + differentPairs - similarPairsErrors - differentPairsErrors) + "\t/\t" +
                    (similarPairs + differentPairs) + "\t=\t" + getAccuracy();
        }
    }
}
