package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.SolutionsDataset;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;

public class TestValidateSplitter {

    public static void main(String[] args) throws IOException {
        split(239566, 148, 148, "min_max");
        split(239566, 200, 200, "double_equality");
        split(239566, 200, 200, "loggers");
        split(239566, 100, 200, "deserialization");
    }

    private static void split(long seed, int validateSize, int testSize, String problem) throws IOException {
        final SolutionsDataset dataset = SolutionsDataset.load(
                Paths.get("cache", "datasets", problem, "reserve.tmp"));
        final List<Solution> solutions = dataset.getValues(x -> x.getVerdict() == FAIL);
        solutions.sort(Comparator.comparingInt(Solution::getSolutionId));
        Collections.shuffle(solutions, new Random(seed));
        final SolutionsDataset validate = new SolutionsDataset(solutions.subList(0, validateSize));
        final SolutionsDataset test = new SolutionsDataset(solutions.subList(validateSize, testSize + validateSize));
        validate.store(Paths.get(".cache", "datasets", problem, "validate.tmp"));
        test.store(Paths.get(".cache", "datasets", problem, "test.tmp"));
    }
}
