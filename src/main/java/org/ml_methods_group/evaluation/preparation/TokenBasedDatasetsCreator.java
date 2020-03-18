package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.HashExtractor;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;

import static org.ml_methods_group.common.Hashers.*;

public class TokenBasedDatasetsCreator {

    private HashMap<String, Integer> dict;
    private List<HashExtractor<CodeChange>> hashers;

    public TokenBasedDatasetsCreator(int wordsLimit, Dataset train,
                                     FeaturesExtractor<Solution, List<Changes>> generator) {
        /*
        List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                .stream()
                .map(generator::process)
                .flatMap(List::stream)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        this.dict = BOWExtractor.mostCommon(hashers, changes, wordsLimit);
        */
        this.hashers = Arrays.asList(getCodeChangeHasher(weak),
                getCodeChangeHasher(javaTypes), getCodeChangeHasher(full), getCodeChangeHasher(extended),
                getCodeChangeHasher(fullExtended), getCodeChangeHasher(deepExtended));
    }

    public void createCodeChangesDataset(List<Solution> solutions,
                                                FeaturesExtractor<Solution, List<Changes>> generator,
                                                Map<Solution, List<String>> marksDictionary,
                                                Path datasetPath) {
        try (var out = new PrintWriter(datasetPath.toFile())) {
            var fullExtractor = hashers.get(2);
            for (Solution solution : solutions) {
                int numberOfNeighbor = 0;
                List<Changes> nearestNeighbors = generator.process(solution);
                for (var item : nearestNeighbors) {
                    out.print(solution.getSolutionId() + Integer.toString(numberOfNeighbor++) + ",");
                    List<CodeChange> changes = item.getChanges();
                    for (CodeChange change : changes) {
                        out.print(fullExtractor.process(change) + ",");
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

    public void createBowDataset(List<Solution> train,
                                        FeaturesExtractor<Solution, List<Changes>> generator,
                                        Map<Solution, List<String>> marksDictionary,
                                        Path datasetPath) {
        BOWExtractor<CodeChange> extractor = new BOWExtractor<>(dict, hashers);
        try (var out = new PrintWriter(datasetPath.toFile())) {
            // Header
            out.print("id,");
            for (int i = 0; i < dict.size(); ++i) {
                out.print(i + ",");
            }
            out.println("cluster");
            // Body
            for (Solution solution : train) {
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
