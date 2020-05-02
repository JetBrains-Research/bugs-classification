package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.evaluation.EvaluationInfo;
import org.ml_methods_group.parsing.JavaCodeValidator;
import org.ml_methods_group.parsing.ParsingUtils;

import java.io.IOException;
import java.io.InputStream;
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
        Set<Integer> problemIds = full.getValues().stream()
                .map(Solution::getProblemId)
                .collect(Collectors.toSet());

        List<Solution> solutions = full.getValues(x -> x.getProblemId() == problemId);
        Map<Integer, List<Solution>> sessionById = new HashMap<>();
        for (Solution solution : solutions) {
            sessionById.computeIfAbsent(solution.getSessionId(), x -> new ArrayList<>())
                    .add(solution);
        }
        List<List<Solution>> sessions = new ArrayList<>(sessionById.values());
        Collections.shuffle(sessions, random);
        List<Solution> train = new ArrayList<>();
        List<Solution> test = new ArrayList<>();
        for (var session : sessions) {
            if (session.size() == 1) continue;
            Solution correct = session.stream().filter(x -> x.getVerdict() == OK)
                    .collect(Collectors.toList()).get(0);
            Solution incorrect = session.stream().filter(x -> x.getVerdict() == FAIL)
                    .collect(Collectors.toList()).get(0);
            if (test.size() < 2 * testSize) {
                test.add(incorrect);
                test.add(correct);
            } else {
                train.add(incorrect);
                train.add(correct);
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
