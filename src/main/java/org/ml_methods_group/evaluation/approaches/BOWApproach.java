package org.ml_methods_group.evaluation.approaches;

import org.ml_methods_group.common.*;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeContext;
import org.ml_methods_group.common.ast.changes.CodeChange.NodeState;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.BOWExtractor.BOWVector;
import org.ml_methods_group.common.extractors.HashExtractor;
import org.ml_methods_group.evaluation.EvaluationInfo;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.serialization.ProtobufSerializationUtils.loadSolutionMarksHolder;

public class BOWApproach {
    public static final HashExtractor<ChangeType> CHANGE_TYPE_HASH = HashExtractor.<ChangeType>builder()
            .append("CT")
            .hashComponent(ChangeType::ordinal)
            .build();


    public static final HashExtractor<NodeType> NODE_TYPE_HASH = HashExtractor.<NodeType>builder()
            .append("NT")
            .hashComponent(NodeType::ordinal)
            .build();

    public static final HashExtractor<NodeState> TYPE_ONLY_NODE_STATE_HASH = HashExtractor.<NodeState>builder()
            .append("TOS")
            .hashComponent(NodeState::getType, NODE_TYPE_HASH)
            .build();

    public static final HashExtractor<NodeState> LABEL_NODE_STATE_HASH = HashExtractor.<NodeState>builder()
            .append("LNS")
            .hashComponent(NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(NodeState::getLabel)
            .build();

    public static final HashExtractor<NodeState> JAVA_TYPE_NODE_STATE_HASH = HashExtractor.<NodeState>builder()
            .append("JNS")
            .hashComponent(NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(NodeState::getJavaType)
            .build();

    public static final HashExtractor<NodeState> FULL_NODE_STATE_HASH = HashExtractor.<NodeState>builder()
            .append("FNS")
            .hashComponent(NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(NodeState::getLabel)
            .hashComponent(NodeState::getJavaType)
            .build();

    public static final ApproachTemplate<BOWVector> TEMPLATE = (d, g) -> getDefaultApproach(20000, d, g);

    public static Approach<BOWVector> getDefaultApproach(int wordsLimit, Dataset train,
                                              FeaturesExtractor<Solution, Changes> generator) {
        final HashExtractor<NodeContext> weak = HashExtractor.<NodeContext>builder()
                .append("TOC")
                .hashComponent(NodeContext::getNode, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> javaTypes = HashExtractor.<NodeContext>builder()
                .append("JTC")
                .hashComponent(NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> full = HashExtractor.<NodeContext>builder()
                .append("FCC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> extended = HashExtractor.<NodeContext>builder()
                .append("ECC")
                .hashComponent(NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> fullExtended = HashExtractor.<NodeContext>builder()
                .append("FEC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> deepExtended = HashExtractor.<NodeContext>builder()
                .append("DEC")
                .hashComponent(NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, LABEL_NODE_STATE_HASH)
                .build();
        return getApproach(wordsLimit, train, generator, Arrays.asList(getCodeChangeHasher(weak),
                getCodeChangeHasher(javaTypes), getCodeChangeHasher(full), getCodeChangeHasher(extended),
                getCodeChangeHasher(fullExtended), getCodeChangeHasher(deepExtended)));
    }

    public static HashExtractor<CodeChange> getCodeChangeHasher(HashExtractor<NodeContext> extractor) {
        return HashExtractor.<CodeChange>builder()
                .append("CCE")
                .hashComponent(CodeChange::getChangeType, CHANGE_TYPE_HASH)
                .hashComponent(CodeChange::getOriginalContext, extractor)
                .hashComponent(CodeChange::getDestinationContext, extractor)
                .build();
    }

    private static Approach<BOWVector> getApproach(int wordsLimit, Dataset train,
                                        FeaturesExtractor<Solution, Changes> generator,
                                        List<HashExtractor<CodeChange>> extractors) {
        final List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                .stream()
                .map(generator::process)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final HashMap<String, Integer> dict = BOWExtractor.mostCommon(extractors, changes, wordsLimit);
        final FeaturesExtractor<Solution, BOWVector> extractor = generator.compose(
                new BOWExtractor<>(dict, extractors).extend(Changes::getChanges));
        return new Approach<>(extractor, BOWExtractor::cosineDistance, "BOW" + wordsLimit);
    }

    public static void createCSV(int wordsLimit,
                                  List<Solution> solutions,
                                  FeaturesExtractor<Solution, Changes> generator,
                                  Map<Solution, List<String>> marksDictionary,
                                  Path datasetPath) throws IOException {
        final HashExtractor<NodeContext> weak = HashExtractor.<NodeContext>builder()
                .append("TOC")
                .hashComponent(NodeContext::getNode, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> javaTypes = HashExtractor.<NodeContext>builder()
                .append("JTC")
                .hashComponent(NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> full = HashExtractor.<NodeContext>builder()
                .append("FCC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> extended = HashExtractor.<NodeContext>builder()
                .append("ECC")
                .hashComponent(NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> fullExtended = HashExtractor.<NodeContext>builder()
                .append("FEC")
                .hashComponent(NodeContext::getNode, FULL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
                .build();
        final HashExtractor<NodeContext> deepExtended = HashExtractor.<NodeContext>builder()
                .append("DEC")
                .hashComponent(NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParent, LABEL_NODE_STATE_HASH)
                .hashComponent(NodeContext::getParentOfParent, LABEL_NODE_STATE_HASH)
                .build();
        List<HashExtractor<CodeChange>> extractors = Arrays.asList(getCodeChangeHasher(weak),
                getCodeChangeHasher(javaTypes), getCodeChangeHasher(full), getCodeChangeHasher(extended),
                getCodeChangeHasher(fullExtended), getCodeChangeHasher(deepExtended));

        final List<CodeChange> allChanges = solutions.stream()
                .map(generator::process)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        final HashMap<String, Integer> dict = BOWExtractor.mostCommon(extractors, allChanges, wordsLimit);
        final FeaturesExtractor<Solution, BOWVector> extractor = generator.compose(
                new BOWExtractor<>(dict, extractors).extend(Changes::getChanges));
        try (var out = new PrintWriter(datasetPath.toFile())) {
            // Header
            out.print("id");
            for (int i = 0; i < dict.size(); ++i) {
                out.print("," + i);
            }
            out.println(",cluster");
            // Body
            for (Solution solution : solutions) {
                out.print(solution.getSolutionId() + ",");
                for (var counter : extractor.process(solution).getCounters()) {
                    out.print(counter + ",");
                }
                var marks = marksDictionary.getOrDefault(solution, new ArrayList<String>());
                if (marks.size() == 0 || marks.size() == 1 && marks.get(0).equals("")) {
                    out.println("unknown");
                    continue;
                }
                for (int i = 0; i < marks.size(); ++i) {
                    if (i == marks.size() - 1)
                        out.println(marks.get(i));
                    else
                        out.print(marks.get(i) + "|");
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
