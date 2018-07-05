package org.ml_methods_group.database;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.preparation.LabelType;
import org.ml_methods_group.core.vectorization.LabelWrapper;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.primitives.Table;
import org.ml_methods_group.database.primitives.Tables;

import java.util.HashMap;
import java.util.Map;

public class StandardLabelIndex implements Index<String, LabelType> {

    private final Table index;

    public StandardLabelIndex(Database database) {
        database.createTable(Tables.STANDARD_LABEL_HEADER);
        this.index = database.getTable(Tables.STANDARD_LABEL_HEADER);
    }

    public StandardLabelIndex() {
        this(new Database());
    }

    @Override
    public void insert(String value, LabelType type) {
        index.insert(new Object[]{value, type});
    }

    @Override
    public Map<String, LabelType> getIndex() {
        final Map<String, LabelType> result = new HashMap<>();
        for (Table.ResultWrapper wrapper : index) {
            result.put(wrapper.getStringValue("label"),
                    wrapper.getEnumValue("type", LabelType.class));
        }
        return result;
    }

    @Override
    public void clean() {
        index.clean();
    }
}
