package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.entities.Solution.Verdict;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.IntPredicate;

import static org.ml_methods_group.core.entities.Solution.Verdict.FAIL;
import static org.ml_methods_group.core.entities.Solution.Verdict.OK;

public class PreparationUtils {
    public static void parse(InputStream stream, JavaCodeValidator validator, IntPredicate problemFilter,
                                Repository<Solution> storage) throws IOException {
        CSVParser<Column> parser = new CSVParser<>(stream, Column::byName, Column.class);
        final HashMap<Long, Integer> sessionIds = new HashMap<>();
        final Map<Long, Solution> lastSolution = new HashMap<>();
        final Map<Long, Long> lastTime = new HashMap<>();
        while (parser.hasNextLine()) {
            parser.nextLine();
            final int problemId = parser.getInt(Column.PROBLEM_ID);
            if (!problemFilter.test(problemId)) {
                continue;
            }
            final Optional<String> code = validator.validate(parser.getToken(Column.CODE));
            if (!code.isPresent()) {
                continue;
            }
            final int userId = parser.getInt(Column.USER_ID);
            final Verdict verdict = parser.getBoolean(Column.VERDICT) ? OK : FAIL;
            final long id = (((long) userId) << 32) | (problemId << 1) | verdict.ordinal();
            sessionIds.putIfAbsent(id >> 1, sessionIds.size());
            final long time = parser.getLongOrDefault(Column.TIME, Long.MAX_VALUE);
            if (time >= lastTime.getOrDefault(id, Long.MIN_VALUE)) {
                final int sessionId = sessionIds.get(id >> 1);
                final Solution solution = new Solution(
                        code.get(),
                        problemId,
                        sessionId,
                        sessionId * 10 + verdict.ordinal(),
                        verdict);
                lastSolution.put(id, solution);
                lastTime.put(id, time);
            }
        }
        lastSolution.values().forEach(storage::insert);
    }

    private enum Column {
        USER_ID("\\S*user_id\\S*", "\\S*data_id\\S*"),
        PROBLEM_ID("\\S*step_id\\S*"),
        VERDICT("\\S*is_passed\\S*", "\\S*status\\S*"),
        CODE("\\S*submission_code\\S*", "\\S*code\\S*"),
        TIME("\\S*timestamp\\S*");

        final String[] names;

        Column(String... names) {
            this.names = names;
        }

        static Optional<Column> byName(String token) {
            for (Column column : values()) {
                if (Arrays.stream(column.names).anyMatch(token::matches)) {
                    return Optional.of(column);
                }
            }
            return Optional.empty();
        }
    }
}
