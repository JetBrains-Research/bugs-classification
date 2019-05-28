package org.ml_methods_group.evaluation.vectorization;

import org.ml_methods_group.common.Dataset;
import org.ml_methods_group.common.Solution;
import org.ml_methods_group.common.ast.changes.BasicChangeGenerator;
import org.ml_methods_group.common.ast.changes.ChangeGenerator;
import org.ml_methods_group.common.ast.changes.Changes;
import org.ml_methods_group.common.ast.changes.CodeChange;
import org.ml_methods_group.common.ast.generation.CachedASTGenerator;
import org.ml_methods_group.common.ast.normalization.NamesASTNormalizer;
import org.ml_methods_group.common.serialization.ProtobufSerializationUtils;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.ml_methods_group.common.CommonUtils.check;
import static org.ml_methods_group.common.Solution.Verdict.FAIL;
import static org.ml_methods_group.common.Solution.Verdict.OK;

public class ParseAllDiffs {
    public static void main(String[] args) throws IOException {
        final Dataset dataset = ProtobufSerializationUtils.loadDataset(
                Paths.get(".cache", "datasets", "train_dataset.tmp"));
        final List<Solution> incorrect = dataset.getValues(check(Solution::getVerdict, FAIL::equals));
        final Map<Integer, Solution> correct = dataset.getValues(check(Solution::getVerdict, OK::equals))
                .stream()
                .collect(Collectors.toMap(Solution::getSessionId, Function.identity()));
        System.out.println("Loaded");
        final ChangeGenerator generator = new BasicChangeGenerator(new CachedASTGenerator(new NamesASTNormalizer()));
        int cnt = 0;
        final Path path = Paths.get(".cache", "out", "train_diffs_sessions.tmp");
        try (PrintWriter output = new PrintWriter(path.toFile())) {
            for (Solution before : incorrect) {
                cnt++;
                if (cnt % 100 == 0) {
                    System.out.println(100.0 * cnt / incorrect.size());
                }

                final Solution after = correct.get(before.getSessionId());
                if (after == null) {
                    continue;
                }
                final Changes changes = generator.getChanges(before, after);
                for (CodeChange change : changes.getChanges()) {
                    SerializationUtils.print(output, change);
                    output.println();
                }
            }
        }
    }
}
