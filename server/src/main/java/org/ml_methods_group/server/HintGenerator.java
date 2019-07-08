package org.ml_methods_group.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.ml_methods_group.classification.classifiers.AdapterClassifier;
import org.ml_methods_group.classification.classifiers.CompositeClassifier;
import org.ml_methods_group.classification.classifiers.KNearestNeighbors;
import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.generation.ASTGenerator;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.extractors.ChangesExtractor;
import org.ml_methods_group.common.metrics.functions.CodeChangeSimilarityMetric;
import org.ml_methods_group.common.metrics.functions.FuzzyJaccardDistanceFunction;
import org.ml_methods_group.common.metrics.functions.HeuristicChangesBasedDistanceFunction;
import org.ml_methods_group.common.metrics.selectors.ClosestPairSelector;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;
import org.ml_methods_group.parsing.CodeValidator;
import org.ml_methods_group.parsing.JavaCodeValidator;

import javax.annotation.Resource;
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
        final var selector = new ClosestPairSelector<>(data.getValues(),
                new HeuristicChangesBasedDistanceFunction(changeGenerator));
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
        System.out.println(generator.getHint(58088, "public static <T> void findMinMax(\n" +
                "    Stream<? extends T> stream,\n" +
                "    Comparator<? super T> order,\n" +
                "    BiConsumer<? super T, ? super T> minMaxConsumer) {\n" +
                "    \n" +
                "    ArrayList<T> list = stream.sorted(order).collect(Collectors.toCollection(ArrayList::new));\n" +
                "    minMaxConsumer.accept(list.get(0), list.get(list.size()-1));\n" +
                "    \n" +
                "}"));
    }
}
