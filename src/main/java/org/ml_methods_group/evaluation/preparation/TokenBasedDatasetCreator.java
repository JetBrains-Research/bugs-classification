package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Hashers.FULL_HASHER;
import static org.ml_methods_group.common.Hashers.getCodeChangeHasher;

public class TokenBasedDatasetCreator implements DatasetCreator {

    public TokenBasedDatasetCreator() {
    }

    @Override
    public void createDataset(List<Solution> solutions,
                              FeaturesExtractor<Solution, List<Changes>> generator,
                              Map<Solution, List<String>> marksDictionary,
                              Path datasetPath) {
        long startTime = System.nanoTime();
        Map<Integer, List<Changes>> preprocessedNeighbours = solutions.parallelStream()
                .collect(Collectors.toMap(Solution::getSolutionId, generator::process));
        long endTime = System.nanoTime();
        System.out.println("Time elapsed: " + TimeUnit.NANOSECONDS.toMillis(endTime - startTime));

        int maxTokens = preprocessedNeighbours.values().parallelStream()
                .flatMap(Collection::stream)
                .map(Changes::getChanges)
                .mapToInt(List::size)
                .max()
                .orElseThrow(() -> new RuntimeException("Empty tokens sequences"));

        try (var out = new PrintWriter(datasetPath.toFile())) {
            var extractor = getCodeChangeHasher(FULL_HASHER);
            int tokensPerChange = extractor.getTokensCount();
            int tokensLineLength = maxTokens * tokensPerChange;
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
                    changes.stream().map(extractor::process).forEach(out::print);
                    for (int i = changes.size() * tokensPerChange; i < tokensLineLength; ++i) {
                        out.print("<PAD>,");
                    }
                    var marks = marksDictionary.getOrDefault(solution, new ArrayList<>());
                    if (marks.stream().allMatch(String::isBlank)) {
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
