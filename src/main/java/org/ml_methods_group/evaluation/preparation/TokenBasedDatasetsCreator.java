package org.ml_methods_group.evaluation.preparation;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.FeaturesExtractor;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.NodeType;
import org.ml_methods_group.common.ast.changes.ChangeType;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.extractors.BOWExtractor;
import org.ml_methods_group.common.extractors.HashExtractor;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.*;

public class TokenBasedDatasetsCreator {

    private HashMap<String, Integer> dict;
    private List<HashExtractor<CodeChange>> hashers;

    public static final HashExtractor<ChangeType> CHANGE_TYPE_HASH = HashExtractor.<ChangeType>builder()
            .append("CT")
            .hashComponent(ChangeType::ordinal)
            .build();
    public static final HashExtractor<NodeType> NODE_TYPE_HASH = HashExtractor.<NodeType>builder()
            .append("NT")
            .hashComponent(NodeType::ordinal)
            .build();
    public static final HashExtractor<CodeChange.NodeState> TYPE_ONLY_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("TOS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeState> LABEL_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("LNS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(CodeChange.NodeState::getLabel)
            .build();
    public static final HashExtractor<CodeChange.NodeState> JAVA_TYPE_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("JNS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .hashComponent(CodeChange.NodeState::getJavaType)
            .build();
    public static final HashExtractor<CodeChange.NodeState> FULL_NODE_STATE_HASH = HashExtractor.<CodeChange.NodeState>builder()
            .append("FNS")
            .hashComponent(CodeChange.NodeState::getType, NODE_TYPE_HASH)
            .append(",")
            .hashComponent(CodeChange.NodeState::getLabel)
            .append(",")
            .hashComponent(CodeChange.NodeState::getJavaType)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> weak = HashExtractor.<CodeChange.NodeContext>builder()
            .append("TOC")
            .hashComponent(CodeChange.NodeContext::getNode, TYPE_ONLY_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> javaTypes = HashExtractor.<CodeChange.NodeContext>builder()
            .append("JTC")
            .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> full = HashExtractor.<CodeChange.NodeContext>builder()
            .append("FCC")
            .hashComponent(CodeChange.NodeContext::getNode, FULL_NODE_STATE_HASH)
            .append(",")
            .hashComponent(CodeChange.NodeContext::getParent, TYPE_ONLY_NODE_STATE_HASH)
            .append(",")
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> extended = HashExtractor.<CodeChange.NodeContext>builder()
            .append("ECC")
            .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> fullExtended = HashExtractor.<CodeChange.NodeContext>builder()
            .append("FEC")
            .hashComponent(CodeChange.NodeContext::getNode, FULL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, TYPE_ONLY_NODE_STATE_HASH)
            .build();
    public static final HashExtractor<CodeChange.NodeContext> deepExtended = HashExtractor.<CodeChange.NodeContext>builder()
            .append("DEC")
            .hashComponent(CodeChange.NodeContext::getNode, JAVA_TYPE_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParent, LABEL_NODE_STATE_HASH)
            .hashComponent(CodeChange.NodeContext::getParentOfParent, LABEL_NODE_STATE_HASH)
            .build();
    public static HashExtractor<CodeChange> getCodeChangeHasher(HashExtractor<CodeChange.NodeContext> hasher) {
        return HashExtractor.<CodeChange>builder()
                .append("CCE")
                .hashComponent(CodeChange::getChangeType, CHANGE_TYPE_HASH)
                .append(",")
                .hashComponent(CodeChange::getOriginalContext, hasher)
                .append(",")
                .hashComponent(CodeChange::getDestinationContext, hasher)
                .build();
    }

    public TokenBasedDatasetsCreator(int wordsLimit, Dataset train,
                                     FeaturesExtractor<Solution, List<Changes>> generator) {
        /*
        List<CodeChange> changes = train.getValues(x -> x.getVerdict() == FAIL)
                .stream()
                .map(generator::process)
                .flatMap(List::stream)
                .map(Changes::getChanges)
                .flatMap(List::stream)
                .collect(Collectors.toList());
        this.dict = BOWExtractor.mostCommon(hashers, changes, wordsLimit);
        */
        this.hashers = Arrays.asList(getCodeChangeHasher(weak),
                getCodeChangeHasher(javaTypes), getCodeChangeHasher(full), getCodeChangeHasher(extended),
                getCodeChangeHasher(fullExtended), getCodeChangeHasher(deepExtended));
    }

    public void createCodeChangesDataset(List<Solution> solutions,
                                                FeaturesExtractor<Solution, List<Changes>> generator,
                                                Map<Solution, List<String>> marksDictionary,
                                                Path datasetPath) {
        try (var out = new PrintWriter(datasetPath.toFile())) {
            var fullExtractor = hashers.get(2);
            for (Solution solution : solutions) {
                int numberOfNeighbor = 0;
                List<Changes> nearestNeighbors = generator.process(solution);
                for (var item : nearestNeighbors) {
                    out.print(solution.getSolutionId() + Integer.toString(numberOfNeighbor++) + ",");
                    List<CodeChange> changes = item.getChanges();
                    for (CodeChange change : changes) {
                        out.print(fullExtractor.process(change) + ",");
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

    public void createBowDataset(List<Solution> train,
                                        FeaturesExtractor<Solution, List<Changes>> generator,
                                        Map<Solution, List<String>> marksDictionary,
                                        Path datasetPath) {
        BOWExtractor<CodeChange> extractor = new BOWExtractor<>(dict, hashers);
        try (var out = new PrintWriter(datasetPath.toFile())) {
            // Header
            out.print("id,");
            for (int i = 0; i < dict.size(); ++i) {
                out.print(i + ",");
            }
            out.println("cluster");
            // Body
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
