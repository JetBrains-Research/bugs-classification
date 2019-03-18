package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.serialization.SolutionsDataset;

import java.io.IOException;
import java.nio.file.Paths;

public class NamedDatasetsCreator {
    public static void main(String[] args) throws IOException {
        SolutionsDataset train = SolutionsDataset.load(Paths.get(".cache","datasets", "train_dataset.tmp"));
        SolutionsDataset test = SolutionsDataset.load(Paths.get(".cache","datasets", "test_dataset.tmp"));

        extract(58088, "min_max", train, test);
        extract(55715, "deserialization", train, test);
        extract(47538, "double_equality", train, test);
        extract(53619, "loggers", train, test);
        extract(57792, "pair", train, test);
        extract(47347, "merge", train, test);
        extract(47334, "factorial", train, test);
        extract(49971, "complex", train, test);
        extract(55689, "double_sum", train, test);
    }


    public static void extract(int problemId, String name,
                               SolutionsDataset train, SolutionsDataset test) throws IOException {
        train.filter(solution -> solution.getProblemId() == problemId)
                .store(Paths.get(".cache","datasets", name, "train.tmp"));
        test.filter(solution -> solution.getProblemId() == problemId)
                .store(Paths.get(".cache","datasets", name, "reserved.tmp"));


    }
}
