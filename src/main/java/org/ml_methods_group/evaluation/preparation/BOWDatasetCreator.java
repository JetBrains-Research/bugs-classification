package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Hashers;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.BOWExtractor;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;


public class BOWDatasetCreator implements DatasetCreator {

    private HashMap<String, Integer> dict;
    private BOWExtractor<CodeChange> extractor;

    public BOWDatasetCreator(List<Solution> allPossibleSolutions,
                             FeaturesExtractor<Solution, List<Changes>> generator,
                             int wordsLimit) {
        List<CodeChange> possibleChanges = allPossibleSolutions.stream()
                .map(generator::process)
                .flatMap(List::stream)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        this.dict = BOWExtractor.mostCommon(Hashers.CODE_CHANGE_HASHERS, possibleChanges, wordsLimit);
        this.extractor = new BOWExtractor<>(dict, Hashers.CODE_CHANGE_HASHERS);
    }

    @Override
    public void createDataset(List<Solution> solutions,
                                 FeaturesExtractor<Solution, List<Changes>> generator,
                                 Map<Solution, List<String>> marksDictionary,
                                 Path pathToDataset) {
        try (var out = new PrintWriter(pathToDataset.toFile())) {
            // CSV header
            out.print("id,");
            for (int i = 0; i < dict.size(); ++i) {
                out.print(i + ",");
            }
            out.println("cluster");
            // Content
            for (Solution solution : solutions) {
                List<Changes> neighbors = generator.process(solution);
                int additionalId = 0;
                for (var item : neighbors) {
                    out.print(solution.getSolutionId() + Integer.toString(additionalId++) + ",");
                    for (var counter : extractor.process(item.getChanges()).getCounters()) {
                        out.print(counter + ",");
                    }
                    var marks = marksDictionary.getOrDefault(solution, new ArrayList<String>());
                    if (marks.stream().allMatch(Objects::isNull)
                            || marks.size() == 1 && marks.get(0).equals("")) {
                        out.println("unknown");
                    } else {
                        out.println(String.join("|", marks));
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
