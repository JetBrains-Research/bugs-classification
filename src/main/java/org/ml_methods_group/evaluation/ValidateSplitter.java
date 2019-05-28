package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.SolutionMarksHolder;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.evaluation.EvaluationInfo.PATH_TO_DATASET;

public class ValidateSplitter {

    public static void main(String[] args) throws IOException {
        int k = 10;
        split(23, k, "factorial");
        split(76, k, "deserialization");
        split(3245, k, "reflection");
        split(65, k, "min_max");
        split(89, k, "double_equality");
        split(123, k, "loggers");
        split(924, k, "pair");
        split(812, k, "merge");
        split(523, k, "read_as_string");
    }

    private static void split(long seed, int k, String problem) throws IOException {
        try {
            final SolutionMarksHolder holder = ProtobufSerializationUtils.loadSolutionMarksHolder(
                    PATH_TO_DATASET.resolve(problem).resolve("train_marks.tmp"));
            final Dataset dataset = ProtobufSerializationUtils.loadDataset(
                    PATH_TO_DATASET.resolve(problem).resolve("train.tmp"));
            final List<Integer> sessions = holder.getSolutions().stream()
                    .map(Solution::getSessionId)
                    .collect(Collectors.toList());
            Collections.shuffle(sessions, new Random(seed));
            final int n = holder.size() / k;
            for (int i = 0; i < k; i++) {
                final Set<Integer> ids = new HashSet<>(sessions.subList(i * n, (i + 1) * n));
                final Dataset validate = dataset.filter(x -> ids.contains(x.getSessionId()));
                final Dataset train = dataset.filter(x -> !ids.contains(x.getSessionId()));
                ProtobufSerializationUtils.storeDataset(validate, PATH_TO_DATASET.resolve(problem).resolve("validation")
                        .resolve("step_" + i).resolve("validate.tmp"));
                ProtobufSerializationUtils.storeDataset(train, PATH_TO_DATASET.resolve(problem).resolve("validation")
                        .resolve("step_" + i).resolve("train.tmp"));
            }
        } catch (FileNotFoundException ignored) { // nothing to split
        }
    }
}
