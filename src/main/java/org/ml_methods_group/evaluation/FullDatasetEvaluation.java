package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.loadDataset;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.storeDataset;

public class FullDatasetEvaluation {

    public static void main(String[] argv) throws Exception {
        final Dataset full = loadDataset(EvaluationInfo.PATH_TO_DATASET.resolve("dataset1.tmp"));
        Set<Integer> problemIds = full.getValues().stream()
                .map(Solution::getProblemId)
                .collect(Collectors.toSet());
        System.out.println("Total solutions: " + full.getValues().size());
        System.out.println("Problems: " + problemIds.size());
        problemIds.forEach(x -> System.out.print(x + " "));

        problemIds.forEach(id -> {
            List<Solution> solutions = full.getValues(x -> x.getProblemId() == id);
            int ratio = (int) (solutions.size() * 0.8);
            List<Solution> train = solutions.subList(0, ratio);
            List<Solution> test = solutions.subList(ratio, solutions.size());
            try {
                Path pathToDataset = EvaluationInfo.PATH_TO_DATASET.resolve(id.toString());
                storeDataset(new Dataset(train), pathToDataset.resolve("train.tmp"));
                storeDataset(new Dataset(test), pathToDataset.resolve("test.tmp"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

}
