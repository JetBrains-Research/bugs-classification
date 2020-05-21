package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.loadDataset;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.storeDataset;

public class FullDatasetEvaluation {

    public static void main(String[] argv) throws Exception {
        final Dataset full = loadDataset(EvaluationInfo.PATH_TO_DATASET.resolve("dataset1.tmp"));
        Set<Integer> problemIds = full.getValues().stream()
                .map(Solution::getProblemId)
                .collect(Collectors.toSet());

        List<String> problems = new ArrayList<>();
        problemIds.forEach(id -> {
            List<Solution> correct = full.getValues(x -> x.getProblemId() == id && x.getVerdict() == OK);
            List<Solution> incorrect = full.getValues(x -> x.getProblemId() == id && x.getVerdict() == FAIL);
            int ratio = (int) Math.round(incorrect.size() * 0.8);
            List<Solution> trainIncorrect = incorrect.subList(0, ratio);
            List<Solution> test = incorrect.subList(ratio, incorrect.size());
            if (trainIncorrect.size() > 400) {
                problems.add(id.toString());
                List<Solution> train = Stream.concat(trainIncorrect.stream(), correct.stream())
                        .collect(Collectors.toList());
                System.out.println(train.size() + " - " + test.size());
                try {
                    Path pathToDataset = EvaluationInfo.PATH_TO_DATASET.resolve("full").resolve(id.toString());
                    storeDataset(new Dataset(train), pathToDataset.resolve("train.tmp"));
                    storeDataset(new Dataset(test), pathToDataset.resolve("test.tmp"));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        System.out.println(problems.size());
        MarkingEvaluation.markGlobalClusters(problems);
    }

}
