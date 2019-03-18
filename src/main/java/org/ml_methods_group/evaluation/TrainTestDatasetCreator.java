package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.parsing.JavaCodeValidator;
import org.ml_methods_group.parsing.ParsingUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class TrainTestDatasetCreator {
    public static void main(String[] args) throws IOException {
        double probability = 0.8;
        long seed = 124345;

        final Random random = new Random(seed);
        final SolutionsDataset dataset;
        try (InputStream fis = TrainTestDatasetCreator.class.getResourceAsStream("/dataset2.csv")) {
            dataset = ParsingUtils.parse(fis, new JavaCodeValidator(), x -> true);
        }
        final HashSet<Integer> test = new HashSet<>();
        final HashSet<Integer> train = new HashSet<>();
        final List<Solution> trainList = new ArrayList<>();
        final List<Solution> testList = new ArrayList<>();
        for (Solution solution : dataset.getValues()) {
            final int sessionId = solution.getSessionId();
            if (test.contains(sessionId)) {
                testList.add(solution);
            } else if (train.contains(sessionId)) {
                trainList.add(solution);
            } else {
                if (random.nextDouble() < probability) {
                    train.add(sessionId);
                    trainList.add(solution);
                } else {
                    test.add(sessionId);
                    testList.add(solution);
                }
            }
        }
        new SolutionsDataset(trainList).store(Paths.get(".cache","datasets", "train_dataset.tmp"));
        new SolutionsDataset(testList).store(Paths.get(".cache","datasets", "test_dataset.tmp"));
    }
}
