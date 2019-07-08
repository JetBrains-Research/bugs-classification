package org.ml_methods_group.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.gumtreediff.tree.ITree;
import org.ml_methods_group.classification.classifiers.AdapterClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
import org.ml_methods_group.common.Classifier;
import org.ml_methods_group.common.CommonUtils;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.ASTUtils;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.extractors.HeuristicASTRepresentationExtractor;
import org.ml_methods_group.common.metrics.functions.CodeChangeSimilarityMetric;
import org.ml_methods_group.common.metrics.functions.EditDistance;
import org.ml_methods_group.common.metrics.functions.FuzzyJaccardDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.HeuristicClosestPairSelector;
import org.ml_methods_group.common.preparation.Unifier;
import org.ml_methods_group.common.preparation.basic.BasicUnifier;
import org.ml_methods_group.common.preparation.basic.MinValuePicker;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.parsing.CodeValidator;
import org.ml_methods_group.parsing.JavaCodeValidator;

import javax.inject.Singleton;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.*;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

@Singleton
@Path("/bugs-classification")
public class HintGenerator {

    private final Map<Integer, Classifier<Solution, String>> classifiers = new HashMap<>();
    private final CodeValidator validator = new JavaCodeValidator();
    private final ObjectMapper mapper = new ObjectMapper();

    public HintGenerator() {
        try (InputStream data = HintGenerator.class.getResourceAsStream("/data.txt");
             Scanner reader = new Scanner(data)) {
            while (reader.hasNext()) {
                final var problemId = Integer.parseInt(reader.nextLine().trim());
                final var marksPath = reader.nextLine().trim();
                final var dataPath = reader.nextLine().trim();
                classifiers.put(problemId, loadClassifier(marksPath, dataPath));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static Classifier<Solution, String> loadClassifier(String markedDataset, String dataset) throws IOException {
        final var data = ProtobufSerializationUtils.loadDataset(Paths.get(dataset))
                .filter(CommonUtils.check(Solution::getVerdict, OK::equals));
        final var marks = ProtobufSerializationUtils.loadMarkedChangesClusters(Paths.get(markedDataset));
        final var treeGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        final var changeGenerator = new BasicChangeGenerator(treeGenerator);
        final Unifier<Solution> unifier = new BasicUnifier<>(
                CommonUtils.compose(treeGenerator::buildTree, ITree::getHash)::apply,
                CommonUtils.checkEquals(treeGenerator::buildTree, ASTUtils::deepEquals),
                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
        final var heuristicExtractor = new HeuristicASTRepresentationExtractor();
        final var selector = new HeuristicClosestPairSelector<>(
                treeGenerator::buildTree,
                new EditDistance(changeGenerator),
                heuristicExtractor,
                heuristicExtractor.getDistanceFunction(),
                unifier.unify(data.getValues()));
        final var metric = CommonUtils.metricFor(
                new FuzzyJaccardDistanceFunction<>(new CodeChangeSimilarityMetric()),
                Changes::getChanges);
        final var changeClassifier = new KNearestNeighbors<Changes, String>(10, metric);
        changeClassifier.train(marks);
        return new AdapterClassifier<>(changeClassifier, new ChangesExtractor(changeGenerator, selector));
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/hint")
    public String getHint(@QueryParam("problem") int problemId,
                          @QueryParam("code") String code) throws IOException {
        final long requestTime = System.currentTimeMillis();
        try {
            final var classifier = classifiers.get(problemId);
            if (classifier == null) {
                return asJSON(Response.error("Unsupported problem!", requestTime));
            }
            final var solution = asSolution(code, problemId);
            if (solution.isEmpty()) {
                return asJSON(Response.error("Failed to build AST", requestTime));
            }
            final var result = classifier.reliability(solution.get());
            return asJSON(Response.success(result, requestTime));

        } catch (Exception e) {
            return asJSON(Response.error(
                    "Unexpected exception: " + e.getClass().getName() + " " + e.getMessage(),
                    requestTime));
        }
    }

    private Optional<Solution> asSolution(String text, int problemId) {
        return validator.validate(text)
                .map(code -> new Solution(code, problemId, -1, -1, FAIL));
    }

    private String asJSON(Object object) throws IOException {
        return mapper.writeValueAsString(object);
    }

    public static void main(String[] args) throws IOException {
        HintGenerator generator = new HintGenerator();
//        final var data = ProtobufSerializationUtils.loadDataset(
//                Paths.get("C:\\internship\\bugs-classification\\data\\53619\\solutions.tmp"))
//                .filter(CommonUtils.check(Solution::getVerdict, OK::equals));
//        final var marks = ProtobufSerializationUtils.loadMarkedChangesClusters(
//                Paths.get("C:\\internship\\bugs-classification\\data\\53619\\marks.tmp"));
//        final var treeGenerator = new CachedASTGenerator(new NamesASTNormalizer());
//        final var changeGenerator = new BasicChangeGenerator(treeGenerator);
//        final Unifier<Solution> unifier = new BasicUnifier<>(
//                CommonUtils.compose(treeGenerator::buildTree, ITree::getHash)::apply,
//                CommonUtils.checkEquals(treeGenerator::buildTree, ASTUtils::deepEquals),
//                new MinValuePicker<>(Comparator.comparingInt(Solution::getSolutionId)));
//        final var heuristicExtractor = new HeuristicASTRepresentationExtractor();
//        final var selector = new HeuristicClosestPairSelector<>(
//                treeGenerator::buildTree,
//                new EditDistance(changeGenerator),
//                heuristicExtractor,
//                heuristicExtractor.getDistanceFunction(),
//                unifier.unify(data.getValues()));
//        System.out.println(data.getValues().size());
//        System.out.println(unifier.unify(data.getValues()).size());
//        final var metric = CommonUtils.metricFor(
//                new FuzzyJaccardDistanceFunction<>(new CodeChangeSimilarityMetric()),
//                Changes::getChanges);
//        final var changeClassifier = new KNearestNeighbors<Changes, String>(10, metric);
//        var changeExtractor = new ChangesExtractor(changeGenerator, selector);
//        changeClassifier.train(marks);
//        Solution solution = generator.asSolution(
//                "private static void configureLogging() {\n" +
//                        "        Logger logA = Logger.getLogger(\"org.stepic.java.logging.ClassA\");\n" +
//                        "        logA.setLevel(Level.ALL);\n" +
//                        "        Logger logB = Logger.getLogger(\"org.stepic.java.logging.ClassB\");\n" +
//                        "        logB.setLevel(Level.WARNING);\n" +
//                        "        Logger logC = Logger.getLogger(\"org.stepic.java\");\n" +
//                        "        ConsoleHandler ch = new ConsoleHandler();\n" +
//                        "        ch.setLevel(Level.ALL);\n" +
//                        "        XMLFormatter formatter = new XMLFormatter();\n" +
//                        "        ch.setFormatter(formatter);\n" +
//                        "        logC.addHander(ch);\n" +
//                        "        logC.setUseParentHandlers(false);\n" +
//                        "    }",
//                53619).get();
//        System.out.println(selector.selectOption(solution).get().getCode());
//        changeExtractor.process(solution).getChanges().forEach(System.out::println);
        for (int i = 0; i < 10; i++) {
            System.out.println(generator.getHint(53619,
                    "private static void configureLogging() {\n" +
                            "        Logger logA = Logger.getLogger(\"org.stepic.java.logging.ClassA\");\n" +
                            "        logA.setLevel(Level.ALL);\n" +
                            "        Logger logB = Logger.getLogger(\"org.stepic.java.logging.ClassB\");\n" +
                            "        logB.setLevel(Level.WARNING);\n" +
                            "        Logger logC = Logger.getLogger(\"org.stepic.java\");\n" +
                            "        ConsoleHandler ch = new ConsoleHandler();\n" +
                            "        ch.setLevel(Level.ALL);\n" +
                            "        XMLFormatter formatter = new XMLFormatter();\n" +
                            "        ch.setFormatter(formatter);\n" +
                            "        logC.addHander(ch);\n" +
                            "        logC.setUseParentHandlers(false);\n" +
                            "    }"));
        }

    }
}
