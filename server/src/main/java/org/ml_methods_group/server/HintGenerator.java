package org.ml_methods_group.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.gumtreediff.matchers.CompositeMatchers;
import com.github.gumtreediff.matchers.MappingStore;
import com.github.gumtreediff.matchers.Matcher;
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
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.MediaType;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.BiFunction;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

@Singleton
@javax.ws.rs.Path("/bugs-classification")
public class HintGenerator {
    private final Map<Integer, Classifier<Solution, String>> classifiers = new HashMap<>();
    private final CodeValidator validator = new JavaCodeValidator();
    private final ObjectMapper mapper = new ObjectMapper();

    public HintGenerator() throws IOException {
        final Path path = Paths.get(new String(
                HintGenerator.class.getResourceAsStream("/data.txt").readAllBytes(),
                Charset.defaultCharset()).trim());
        final String[] data = path.toFile()
                .list((dir, name) -> name.matches("\\d{1,9}"));
        if (data == null) {
            throw new IOException("Data folder wasn't found!");
        }
        for (var problem : data) {
            try {
                final var problemId = Integer.parseInt(problem);
                final var marksPath = path.resolve(problem).resolve("prepared.tmp");
                final var dataPath = path.resolve(problem).resolve("solutions.tmp");
                classifiers.put(problemId, loadClassifier(marksPath, dataPath));
            } catch (Exception e) {
                throw new IOException(e);
            }
        }
    }

    private static Classifier<Solution, String> loadClassifier(Path markedDataset, Path dataset) throws IOException {
        final var data = ProtobufSerializationUtils.loadDataset(dataset)
                .filter(CommonUtils.check(Solution::getVerdict, OK::equals));
        final var marks = ProtobufSerializationUtils.loadMarkedChangesClusters(markedDataset);
        final var treeGenerator = new CachedASTGenerator(new NamesASTNormalizer());
        final var changeGenerator = new BasicChangeGenerator(treeGenerator,
                Collections.singletonList((Serializable & BiFunction<ITree, ITree, Matcher>) (x, y) ->
                        new CompositeMatchers.ClassicGumtree(x, y, new MappingStore())));
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
        final var changeClassifier = new KNearestNeighbors<Changes, String>(5, metric);
        changeClassifier.train(marks);
        return new AdapterClassifier<>(changeClassifier, new ChangesExtractor(changeGenerator, selector));
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @javax.ws.rs.Path("/classifiers")
    public String getClassifiers() throws IOException {
        final var response = new ArrayList<>(classifiers.keySet());
        return asJSON(response);
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @javax.ws.rs.Path("/hint")
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
        System.out.println(generator.getHint(53676, "public static String getCallerClassAndMethodName() {\n" +
                "        try {\n" +
                "            throw new Exception(\"test\");\n" +
                "        } catch (Exception e) {\n" +
                "            StackTraceElement[] stackTraceElements = e.getStackTrace();\n" +
                "            if (stackTraceElements.length >= 3) {\n" +
                "                return stackTraceElements[1].getClassName() + \"#\" + stackTraceElements[1].getMethodName();\n" +
                "            }\n" +
                "        }\n" +
                "        return null;\n" +
                "}"));
    }
}
