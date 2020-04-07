package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public interface DatasetCreator {
    void createDataset(List<Solution> solutions,
                       FeaturesExtractor<Solution, List<Changes>> generator,
                       Map<Solution, List<String>> marksDictionary,
                       Path pathToDataset);
}
