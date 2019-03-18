package org.ml_methods_group.evaluation;

import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.testing.validation.basic.PrecalculatedValidator;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;

public class ValidatorCreator {

    public static void main(String[] args) throws IOException {
//        createValidator("min_max");
        createValidator("double_equality");
    }

    private static void createValidator(String problem) throws IOException {
        final List<Solution> incorrect =  new ArrayList<>();
        SolutionsDataset.load(Paths.get("cache", "datasets", problem, "test.tmp"))
                .filter(x -> x.getVerdict() == FAIL)
                .forEach(incorrect::add);
        SolutionsDataset.load(Paths.get("cache", "datasets", problem, "validate.tmp"))
                .filter(x -> x.getVerdict() == FAIL)
                .forEach(incorrect::add);
        final PrecalculatedValidator validator = PrecalculatedValidator.create(incorrect);
        final Path path = Paths.get("cache", "validators", problem + ".pvd");
        path.getParent().toFile().mkdirs();
        validator.store(path);
    }
}
