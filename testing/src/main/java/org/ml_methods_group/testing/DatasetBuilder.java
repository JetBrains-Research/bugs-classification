package org.ml_methods_group.testing;

import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.Solution.Verdict;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.testing.database.ConditionSupplier;
import org.ml_methods_group.testing.database.Repository;

import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.*;

public class DatasetBuilder {

    private final long seed;
    private final List<Solution> correct;
    private final List<Solution> incorrect;

    public DatasetBuilder(long seed, SolutionsDataset dataset) {
        this.seed = seed;
        correct = dataset.getValues(CommonUtils.check(Solution::getVerdict, OK::equals));
        incorrect = dataset.getValues(CommonUtils.check(Solution::getVerdict, FAIL::equals));
        validate(correct, incorrect, seed);
    }

    public int size() {
        return Math.min(correct.size(), incorrect.size());
    }

    public List<Solution> getAllCorrect() {
        return Collections.unmodifiableList(correct);
    }

    public List<Solution> getAllIncorrect() {
        return Collections.unmodifiableList(incorrect);
    }

    public TrainTestSplit createTrainTestSplit(int trainSize, int testSize) {
        final List<Solution> trainIncorrect = incorrect.subList(0, trainSize);
        final List<Solution> testIncorrect = incorrect.subList(trainSize, trainSize + testSize);
        final Set<Integer> testSessions = testIncorrect.stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toSet());
        final List<Solution> trainCorrect = correct.stream()
                .filter(CommonUtils.checkNot(Solution::getSessionId, testSessions::contains))
                .collect(Collectors.toList());
        return new TrainTestSplit(trainCorrect, trainIncorrect, testIncorrect);
    }

    private static List<Solution> loadSolutions(Repository<Solution> repository, int problemId, Verdict verdict) {
        final ConditionSupplier conditions = repository.conditionSupplier();
        final List<Solution> result = new ArrayList<>();
        repository.get(conditions.is("problemid", problemId), conditions.is("verdict", verdict))
                .forEachRemaining(result::add);
        return result;
    }

    private static void validate(List<Solution> correct, List<Solution> incorrect, long seed) {
        final Set<Integer> sessions = correct.stream()
                .map(Solution::getSessionId)
                .collect(Collectors.toSet());
        incorrect.removeIf(x -> !sessions.contains(x.getSessionId()));
        incorrect.sort(Comparator.comparingInt(Solution::getSolutionId));
        Collections.shuffle(incorrect, new Random(seed));
    }

    public long getSeed() {
        return seed;
    }

    public static class TrainTestSplit {
        private final List<Solution> trainCorrect;
        private final List<Solution> trainIncorrect;
        private final List<Solution> testIncorrect;

        private TrainTestSplit(List<Solution> trainCorrect, List<Solution> trainIncorrect, List<Solution> testIncorrect) {
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