package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.evaluation.EvaluationInfo;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.ml_methods_group.evaluation.EvaluationInfo.PATH_TO_DATASET;

public class NamedDatasetsCreator {
    public static void main(String[] args) throws IOException {
        final Dataset train = ProtobufSerializationUtils.loadDataset(PATH_TO_DATASET.resolve("train_dataset.tmp"));
        final Dataset test = ProtobufSerializationUtils.loadDataset(PATH_TO_DATASET.resolve("test_dataset.tmp"));
        extract(47334, "factorial", train, test);
        extract(55715, "deserialization", train, test);
        extract(53676, "reflection", train, test);
        extract(58088, "min_max", train, test);
        extract(47538, "double_equality", train, test);
        extract(53619, "loggers", train, test);
        extract(57792, "pair", train, test);
        extract(47347, "merge", train, test);
        extract(55673, "read_as_string", train, test);
    }


    private static void extract(int problemId, String name,
                               Dataset train, Dataset test) throws IOException {
        ProtobufSerializationUtils.storeDataset(
                train.filter(solution -> solution.getProblemId() == problemId),
                PATH_TO_DATASET.resolve(name).resolve("train.tmp"));
        ProtobufSerializationUtils.storeDataset(
                test.filter(solution -> solution.getProblemId() == problemId),
                PATH_TO_DATASET.resolve(name).resolve("test.tmp"));


    }
}
