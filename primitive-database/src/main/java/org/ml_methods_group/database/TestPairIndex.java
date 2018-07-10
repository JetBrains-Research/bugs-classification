package org.ml_methods_group.database;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.testing.ExternalTester.PairGuess;
import org.ml_methods_group.core.testing.Pair;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.primitives.Table;
import org.ml_methods_group.database.primitives.Tables;

import java.util.HashMap;
import java.util.Map;

public class TestPairIndex implements Index<Pair<Integer>, PairGuess> {

    private final Table index;

    public TestPairIndex(Database database) {
        database.createTable(Tables.TEST_PAIRS_HEADER);
        this.index = database.getTable(Tables.TEST_PAIRS_HEADER);
    }

    public TestPairIndex() {
        this(new Database());
    }

    @Override
    public void insert(Pair<Integer> value, PairGuess guess) {
        index.insert(new Object[]{value.first, value.second, guess});
    }

    @Override
    public Map<Pair<Integer>, PairGuess> getIndex() {
        final Map<Pair<Integer>, PairGuess> result = new HashMap<>();
        for (Table.ResultWrapper wrapper : index) {
            final Pair<Integer> value = new Pair<>(
                    wrapper.getIntValue("first"),
                    wrapper.getIntValue("second"));
            result.put(value, wrapper.getEnumValue("guess", PairGuess.class));
        }
        return result;
    }

    @Override
    public void clean() {
        index.clean();
    }
}
