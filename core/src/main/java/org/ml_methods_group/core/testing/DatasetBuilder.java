package org.ml_methods_group.core.testing;

import org.ml_methods_group.core.CommonUtils;
import org.ml_methods_group.core.database.ConditionSupplier;
import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.entities.Solution.Verdict;

import java.util.*;
import java.util.stream.Collectors;

public class DatasetBuilder {

    private final long seed;
    private final int problemId;
    private final List<Solution> correct;
    private final List<Solution> incorrect;

    public DatasetBuilder(long seed, int problemId, Repository<Solution> repository) {
        this.seed = seed;
        this.problemId = problemId;
        correct = loadSolutions(repository, problemId, Verdict.OK);
        incorrect = loadSolutions(repository, problemId, Verdict.FAIL);
        Collections.shuffle(incorrect, new Random(seed));
    }

    public int size() {
        return Math.min(correct.size(), incorrect.size());
    }

    public Dataset createDataset(int trainSize, int testSize) {
        final List<Solution> trainIncorrect = incorrect.subList(0, trainSize);
        final List<Solution> testIncorrect = incorrect.subList(trainSize, trainSize + testSize);
        final Set<Integer> testSessions = testIncorrect.stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toSet());
        final List<Solution> trainCorrect = correct.stream()
                .filter(CommonUtils.checkNot(Solution::getSessionId, testSessions::contains))
                .collect(Collectors.toList());
        return new Dataset(trainCorrect, trainIncorrect, testIncorrect);
    }

    private static List<Solution> loadSolutions(Repository<Solution> repository, int problemId, Verdict verdict) {
        final ConditionSupplier conditions = repository.conditionSupplier();
        final List<Solution> result = new ArrayList<>();
        repository.get(conditions.is("problemid", problemId), conditions.is("verdict", verdict))
                .forEachRemaining(result::add);
        result.sort(Comparator.comparingInt(Solution::getSolutionId));
        return result;
    }

    public long getSeed() {
        return seed;
    }

    public int getProblemId() {
        return problemId;
    }

    public static class Dataset {
        private final List<Solution> trainCorrect;
        private final List<Solution> trainIncorrect;
        private final List<Solution> testIncorrect;

        private Dataset(List<Solution> trainCorrect, List<Solution> trainIncorrect, List<Solution> testIncorrect) {
            this.trainCorrect = trainCorrect;
            this.trainIncorrect = trainIncorrect;
            this.testIncorrect = testIncorrect;
        }

        public List<Solution> getTrainCorrect() {
            return trainCorrect;
        }

        public List<Solution> getTrainIncorrect() {
            return trainIncorrect;
        }

        public List<Solution> getTestIncorrect() {
            return testIncorrect;
        }
    }
}
