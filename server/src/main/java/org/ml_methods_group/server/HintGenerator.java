package org.ml_methods_group.server;

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

import javax.annotation.Resource;
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

@Path("/bugs-classification")
public class HintGenerator {

    private final Map<Integer, Classifier<Solution, String>> classifiers = new HashMap<>();

    public HintGenerator() {
        try (InputStream data = HintGenerator.class.getResourceAsStream("/meta.txt");
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
        final var changeClassifier = new KNearestNeighbors<Changes, String>(15, metric);
        changeClassifier.train(marks);
        return new AdapterClassifier<>(changeClassifier, new ChangesExtractor(changeGenerator, selector));
    }

    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Path("/hint")
    public Response getHint(@QueryParam("problem") int problemId,
                            @QueryParam("code") String code) {
        try {
            final var classifier = classifiers.get(problemId);
            if (classifier == null) {
                return Response.error("Unsupported problem!");
            }
            final var solution = new Solution(code, problemId, -1, -1, FAIL);
            final var result = classifier.mostProbable(solution);
            return Response.success(result.getKey(), result.getValue());
        } catch (Exception e) {
            return Response.error("Unexpected exception: " + e.getClass().getName() + " " + e.getMessage());
        }
    }
}
