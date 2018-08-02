package org.ml_methods_group.core.preparation;

import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;

import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;
import java.util.function.IntPredicate;

public class PreparationUtils {
    public static void parse(InputStream stream, JavaCodeValidator validator, IntPredicate problemFilter,
                             Repository<Solution> storage) throws IOException {
        CSVParser parser = new CSVParser(stream);
        while (parser.hasNextLine()) {
            parser.nextLine();
            if (!problemFilter.test(parser.getProblemId())) {
                continue;
            }
            final Optional<String> code = validator.validate(parser.getCode());
            if (!code.isPresent()) {
                continue;
            }
            final Solution solution = new Solution(
                    code.get(),
                    parser.getProblemId(),
                    parser.getSessionId(),
                    parser.getSessionId() * 10 + parser.getVerdict().ordinal(),
                    parser.getVerdict());
            storage.insert(solution);
        }
    }
}
