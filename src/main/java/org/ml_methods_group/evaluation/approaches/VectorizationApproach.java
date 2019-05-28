package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.embedding.*;
import org.ml_methods_group.common.extractors.PointwiseExtractor;
import org.ml_methods_group.common.metrics.functions.FunctionsUtils;
import org.ml_methods_group.common.metrics.functions.FuzzyJaccardDistanceFunction;
import org.ml_methods_group.evaluation.vectorization.SerializationUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.embedding.ListEmbeddingExtractor.Reducing.MEAN;
import static org.ml_methods_group.common.embedding.ListEmbeddingExtractor.Reducing.SUM;

public class VectorizationApproach {

    public static Approach<List<double[]>> getDefaultApproach(Dataset train,
                                                              FeaturesExtractor<Solution, Changes> generator) {
        try {
            final Map<String, Integer> wordsDict = SerializationUtils.readMap(".cache/dicts/words_dict.txt");
            final Map<String, Integer> javaTypesDict = SerializationUtils.readMap(".cache/dicts/java_types_dict.txt");
            final EmbeddingExtractor<CodeChange> extractor = createCodeChangeExtractor(
                    ".cache/embeddings/", wordsDict, javaTypesDict);
            final List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                    .stream()
                    .map(generator::process)
                    .map(Changes::getChanges)
                    .flatMap(List::stream)
                    .collect(Collectors.toList());
            final NormalizedVectorExtractor<CodeChange> normalizedExtractor = NormalizedVectorExtractor.normalization(
                    changes, extractor);
            return new Approach<>(generator.compose(Changes::getChanges)
                    .compose(new PointwiseExtractor<>(normalizedExtractor)),
                    new FuzzyJaccardDistanceFunction<>(FunctionsUtils::cosineSimilarity), "def_vec");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Approach<double[]> getSumApproach(Dataset train,
                                                    FeaturesExtractor<Solution, Changes> generator) {
        try {
            final Map<String, Integer> wordsDict = SerializationUtils.readMap(".cache/dicts/words_dict.txt");
            final Map<String, Integer> javaTypesDict = SerializationUtils.readMap(".cache/dicts/java_types_dict.txt");
            final EmbeddingExtractor<CodeChange> extractor = createCodeChangeExtractor(
                    ".cache/embeddings/", wordsDict, javaTypesDict);
            final List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                    .stream()
                    .map(generator::process)
                    .map(Changes::getChanges)
                    .flatMap(List::stream)
                    .collect(Collectors.toList());
            final NormalizedVectorExtractor<CodeChange> normalizedExtractor = NormalizedVectorExtractor.normalization(
                    changes, extractor);
            return new Approach<>(generator.compose(Changes::getChanges)
                    .compose(new ListEmbeddingExtractor<>(normalizedExtractor, SUM)),
                    FunctionsUtils::cosineDistance, "sum_vec");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Approach<double[]> getMeanApproach(Dataset train,
                                                     FeaturesExtractor<Solution, Changes> generator)
            throws FileNotFoundException {
        final Map<String, Integer> wordsDict = SerializationUtils.readMap(".cache/dicts/words_dict.txt");
        final Map<String, Integer> javaTypesDict = SerializationUtils.readMap(".cache/dicts/java_types_dict.txt");
        final EmbeddingExtractor<CodeChange> extractor = createCodeChangeExtractor(
                ".cache/embeddings/", wordsDict, javaTypesDict);
        final List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                .stream()
                .map(generator::process)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final NormalizedVectorExtractor<CodeChange> normalizedExtractor = NormalizedVectorExtractor.normalization(
                changes, extractor);
        return new Approach<>(generator.compose(Changes::getChanges)
                .compose(new ListEmbeddingExtractor<>(normalizedExtractor, MEAN)),
                FunctionsUtils::cosineDistance, "mean_vec");
    }

    private static EmbeddingExtractor<NodeState> createNodeStateExtractor(String name,
                                                                          Map<String, Integer> words,
                                                                          Map<String, Integer> javaTypes) {
        final Embedding<NodeType, Integer> type = Embedding.loadEmbedding(
                new File(name + ".emb"),
                Integer::parseInt,
                NodeType::ordinal,
                99);
        final Embedding<String, Integer> label = Embedding.loadEmbedding(
                new File(name + "_labels.emb"),
                Integer::parseInt,
                words::get,
                0);
        final Embedding<String, Integer> javaType = Embedding.loadEmbedding(
                new File(name + "_java_types.emb"),
                Integer::parseInt,
                javaTypes::get,
                0);
        return new NodeStateEmbeddingExtractor(type, label, javaType);
    }

    private static EmbeddingExtractor<NodeContext> createNodeContextExtractor(String name,
                                                                              Map<String, Integer> words,
                                                                              Map<String, Integer> javaTypes) {
        final EmbeddingExtractor<NodeState> node = createNodeStateExtractor(
                name + "_nodes", words, javaTypes);
        final EmbeddingExtractor<NodeState> parent = createNodeStateExtractor(
                name + "_parents", words, javaTypes);
        final EmbeddingExtractor<NodeState> parentOfParent = createNodeStateExtractor(
                name + "_parent_of_parents", words, javaTypes);

        final EmbeddingExtractor<NodeState> leftUncle = createNodeStateExtractor(
                name + "_left_uncles", words, javaTypes);
        final EmbeddingExtractor<NodeState> rightUncle = createNodeStateExtractor(
                name + "_right_uncles", words, javaTypes);

        final EmbeddingExtractor<NodeState> leftBrother = createNodeStateExtractor(
                name + "_left_brothers", words, javaTypes);
        final EmbeddingExtractor<NodeState> rightBrother = createNodeStateExtractor(
                name + "_right_brothers", words, javaTypes);

        final EmbeddingExtractor<NodeState> child = createNodeStateExtractor(
                name + "_children", words, javaTypes);
        return new NodeContextEmbeddingExtractor(node, parent, parentOfParent,
                new PrefixEmbeddingExtractor<>(child, 6),
                new SequenceContextEmbeddingExtractor<>(leftBrother, rightBrother, 3),
                new SequenceContextEmbeddingExtractor<>(leftUncle, rightUncle, 3));
    }

    private static EmbeddingExtractor<CodeChange> createCodeChangeExtractor(String name,
                                                                            Map<String, Integer> words,
                                                                            Map<String, Integer> javaTypes) {
        final EmbeddingExtractor<NodeContext> originalExtractor = createNodeContextExtractor(
                name + "o", words, javaTypes);
        final EmbeddingExtractor<NodeContext> destinationExtractor = createNodeContextExtractor(
                name + "d", words, javaTypes);
        final Embedding<ChangeType, Integer> changeTypeEmbedding = Embedding.loadEmbedding(
                new File(name + "change_type.emb"),
                Integer::parseInt,
                ChangeType::ordinal,
                0);
        return new ChangeEmbeddingExtractor(changeTypeEmbedding, originalExtractor, destinationExtractor);
    }
}
