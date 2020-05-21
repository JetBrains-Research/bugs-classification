package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.evaluation.EvaluationInfo;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.loadDataset;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.storeDataset;

public class TrainTestDatasetCreator {

    public static void main(String[] args) throws IOException {
        long seed = 124345;
        int problemId = 57810;
        int testSize = 200;

        final Random random = new Random(seed);
        final Dataset full = loadDataset(EvaluationInfo.PATH_TO_DATASET.resolve("dataset2.tmp"));
        List<Solution> solutions = full.getValues(x -> x.getProblemId() == problemId);
        Map<Integer, List<Solution>> sessionById = new HashMap<>();
        for (Solution solution : solutions) {
            sessionById.computeIfAbsent(solution.getSessionId(), x -> new ArrayList<>()).add(solution);
        }
        List<List<Solution>> sessions = new ArrayList<>(sessionById.values());
        Collections.shuffle(sessions, random);
        List<Solution> train = new ArrayList<>();
        List<Solution> test = new ArrayList<>();
        for (var session : sessions) {
            List<Solution> correct = session.stream()
                    .filter(x -> x.getVerdict() == OK)
                    .collect(Collectors.toList());
            List<Solution> incorrect = session.stream()
                    .filter(x -> x.getVerdict() == FAIL)
                    .collect(Collectors.toList());
            long currentTestSize = test.stream().filter(x -> x.getVerdict() == FAIL).count();
            if (currentTestSize < testSize && !incorrect.isEmpty()) {
                test.addAll(incorrect);
                test.addAll(correct);
            } else {
                train.addAll(incorrect);
                train.addAll(correct);
            }
        }

        try {
            Path pathToDataset = EvaluationInfo.PATH_TO_DATASET.resolve("filter");
            storeDataset(new Dataset(train), pathToDataset.resolve("train.tmp"));
            storeDataset(new Dataset(test), pathToDataset.resolve("test.tmp"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
