package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.HashExtractor;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Hashers.*;

public class TokenBasedDatasetsCreator {

    public static void createCodeChangesDataset(List<Solution> solutions,
                                                FeaturesExtractor<Solution, List<Changes>> generator,
                                                Map<Solution, List<String>> marksDictionary,
                                                Path datasetPath) {
        var preprocessedNeighbours = new HashMap<Integer, List<Changes>>();
        var maxTokens = new AtomicInteger(0);
        solutions.parallelStream().forEach(solution -> {
            List<Changes> kChanges = generator.process(solution);
            preprocessedNeighbours.put(solution.getSolutionId(), kChanges);
            int tokensLength = kChanges.stream()
                    .map(x -> x.getChanges().size())
                    .max(Comparator.comparing(Integer::valueOf)).get();
            maxTokens.getAndAccumulate(tokensLength, Math::max);
        });
        try (var out = new PrintWriter(datasetPath.toFile())) {
            var extractor = getCodeChangeHasher(full);
            int tokensPerChange = 11;
            int tokensLineLength = maxTokens.get() * tokensPerChange;
            // CSV header
            out.print("id,real_len,");
            for (int i = 0; i < tokensLineLength; ++i) {
                out.print(i + ",");
            }
            out.println("cluster");
            // Content
            for (Solution solution : solutions) {
                int idSuffix = 0;
                List<Changes> nearestNeighbours = preprocessedNeighbours.get(solution.getSolutionId());
                for (var neighbour : nearestNeighbours) {
                    out.print(solution.getSolutionId() + Integer.toString(idSuffix++) + ",");
                    List<CodeChange> changes = neighbour.getChanges();
                    out.print(changes.size() * tokensPerChange + ",");
                    for (CodeChange cc : changes) {
                        String atomicCodeChangeTokens = extractor.process(cc).replaceAll("[,'\"]", "");
                        out.print(atomicCodeChangeTokens.replace((char)31, ',') + ",");
                    }
                    for (int i = changes.size() * tokensPerChange; i < tokensLineLength; ++i) {
                        out.print("<PAD>,");
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

    public static void createBowDataset(List<Solution> train,
                                        int wordsLimit,
                                        FeaturesExtractor<Solution, List<Changes>> generator,
                                        Map<Solution, List<String>> marksDictionary,
                                        Path datasetPath) {
        List<CodeChange> changes = train.stream()
                .map(generator::process)
                .flatMap(List::stream)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        var dict = BOWExtractor.mostCommon(hashers, changes, wordsLimit);
        BOWExtractor<CodeChange> extractor = new BOWExtractor<>(dict, hashers);
        try (var out = new PrintWriter(datasetPath.toFile())) {
            // CSV header
            out.print("id,");
            for (int i = 0; i < dict.size(); ++i) {
                out.print(i + ",");
            }
            out.println("cluster");
            // Content
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
