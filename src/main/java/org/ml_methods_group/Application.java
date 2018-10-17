package org.ml_methods_group;

import org.ml_methods_group.classification.ClassificationUtils;
import org.ml_methods_group.clustering.ClusterizationUtils;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.serialization.MarkedSolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionClassifier;
import org.ml_methods_group.common.serialization.SolutionsClusters;
import org.ml_methods_group.common.serialization.SolutionsDataset;
import org.ml_methods_group.marking.MarkingUtils;
import org.ml_methods_group.parsing.ParsingUtils;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Collectors;

public class Application {
    public static void main(String[] args) throws IOException {
        final String command = args[0];
        switch (command) {
            case "parse":
                parseDataset(Integer.parseInt(args[1]), args[2], args[3]);
                break;
            case "cluster":
                buildClusters(args[1], args[2]);
                break;
            case "mark":
                markClusters(args[1], args[2]);
                break;
            case "train":
                trainClassifier(args[1], args[2], args[3]);
                break;
            case "classify":
                classify(args[1], args[2]);
                break;
        }

    }

    private static void parseDataset(int problemId, String inputPath, String outputPath) throws IOException {
        final Path output = Paths.get(outputPath);
        try (FileInputStream fis = new FileInputStream(inputPath)) {
            ParsingUtils.parseJavaSolutions(fis, problemId).store(output);
        }
    }

    private static void buildClusters(String inputPath, String outputPath) throws IOException {
        final SolutionsDataset dataset = SolutionsDataset.load(Paths.get(inputPath));
        ClusterizationUtils.buildClusters(dataset).store(Paths.get(outputPath));
    }

    private static void markClusters(String inputPath, String outputPath) throws IOException {
        final SolutionsClusters clusters = SolutionsClusters.load(Paths.get(inputPath));
        MarkingUtils.markClusters(clusters).store(Paths.get(outputPath));
    }

    private static void trainClassifier(String datasetPath, String marksPath, String outputPath) throws IOException {
        final SolutionsDataset dataset = SolutionsDataset.load(Paths.get(datasetPath));
        final MarkedSolutionsClusters clusters = MarkedSolutionsClusters.load(Paths.get(marksPath));
        ClassificationUtils.trainClassifier(clusters, dataset).store(Paths.get(outputPath));
    }

    private static void classify(String classifier, String input) throws IOException {
        final String code = Files.lines(Paths.get(input)).collect(Collectors.joining(System.lineSeparator()));
        final Solution fake = new Solution(code, -1, -1, -1, Solution.Verdict.FAIL);
        final String mark = SolutionClassifier.load(Paths.get(classifier)).classify(fake);
        System.out.println(mark);
    }
}
