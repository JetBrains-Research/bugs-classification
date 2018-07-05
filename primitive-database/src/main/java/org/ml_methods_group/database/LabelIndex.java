package org.ml_methods_group.database;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.vectorization.LabelWrapper;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.primitives.Table;
import org.ml_methods_group.database.primitives.Tables;

import java.util.HashMap;
import java.util.Map;

public class LabelIndex implements Index<LabelWrapper> {

    private final Table index;

    public LabelIndex(Database database) {
        database.createTable(Tables.LABEL_INDEX_HEADER);
        this.index = database.getTable(Tables.LABEL_INDEX_HEADER);
    }

    public LabelIndex() {
        this(new Database());
    }

    @Override
    public void insert(LabelWrapper value, int count) {
        index.insert(new Object[]{value.getLabel(),value.getId(), count});
    }

    @Override
    public Map<LabelWrapper, Integer> getIndex() {
        final Map<LabelWrapper, Integer> result = new HashMap<>();
        for (Table.ResultWrapper wrapper : index) {
            final LabelWrapper value = new LabelWrapper(
                    wrapper.getStringValue("label"),
                    wrapper.getIntValue("id")
            );
            result.put(value, wrapper.getIntValue("count"));
        }
        return result;
    }

    @Override
    public void clean() {
        index.clean();
    }
}
