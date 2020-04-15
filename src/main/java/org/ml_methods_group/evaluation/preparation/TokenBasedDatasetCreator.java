package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.ml_methods_group.common.Hashers.*;

public class TokenBasedDatasetCreator implements DatasetCreator {

    public TokenBasedDatasetCreator() {}

    @Override
    public void createDataset(List<Solution> solutions,
                              FeaturesExtractor<Solution, List<Changes>> generator,
                              Map<Solution, List<String>> marksDictionary,
                              Path datasetPath) {
        var preprocessedNeighbours = new HashMap<Integer, List<Changes>>();
        var maxTokens = new AtomicInteger(0);

        long startTime = System.nanoTime();
        solutions.parallelStream().forEach(solution -> {
            List<Changes> kChanges = generator.process(solution);
            preprocessedNeighbours.put(solution.getSolutionId(), kChanges);
            int tokensLength = kChanges.stream()
                    .map(x -> x.getChanges().size())
                    .max(Comparator.comparing(Integer::valueOf)).get();
            maxTokens.getAndAccumulate(tokensLength, Math::max);
        });
        long endTime = System.nanoTime();
        System.out.println("Time elapsed: " + TimeUnit.NANOSECONDS.toMillis(endTime - startTime));

        try (var out = new PrintWriter(datasetPath.toFile())) {
            var extractor = getCodeChangeHasher(FULL_EXTENDED_HASHER);
            int tokensPerChange = extractor.getTokensCount();
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
                        out.print(extractor.process(cc));
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
}
