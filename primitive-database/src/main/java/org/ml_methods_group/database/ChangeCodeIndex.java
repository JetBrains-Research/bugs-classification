package org.ml_methods_group.database;

import org.ml_methods_group.core.Index;
import org.ml_methods_group.core.changes.ChangeType;
import org.ml_methods_group.core.changes.NodeType;
import org.ml_methods_group.core.vectorization.ChangeCodeWrapper;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.primitives.Table;
import org.ml_methods_group.database.primitives.Tables;

import java.util.HashMap;
import java.util.Map;

public class ChangeCodeIndex implements Index<ChangeCodeWrapper, Integer> {

    private final Table index;

    public ChangeCodeIndex(Database database) {
        database.createTable(Tables.CHANGE_CODE_INDEX_HEADER);
        this.index = database.getTable(Tables.CHANGE_CODE_INDEX_HEADER);
    }

    public ChangeCodeIndex() {
        this(new Database());
    }

    @Override
    public void insert(ChangeCodeWrapper value, Integer count) {
        index.insert(new Object[]{
                value.getCode(),
                value.getEncodingType(),
                count,
                value.getChangeType(),
                value.getNodeType(),
                value.getParentType(),
                value.getParentOfParentType(),
                value.getOldParentType(),
                value.getOldParentOfParentType(),
                value.getLabel(),
                value.getOldLabel()
        });
    }

    @Override
    public Map<ChangeCodeWrapper, Integer> getIndex() {
        final Map<ChangeCodeWrapper, Integer> result = new HashMap<>();
        for (Table.ResultWrapper wrapper : index) {
            final ChangeCodeWrapper value = new ChangeCodeWrapper(
                    wrapper.getBigIntValue("code"),
                    wrapper.getIntValue("encoding_type"),
                    wrapper.getEnumValue("change_type", ChangeType.class),
                    wrapper.getEnumValue("node_type", NodeType.class),
                    wrapper.getEnumValue("parent_type", NodeType.class),
                    wrapper.getEnumValue("parent_of_parent_type", NodeType.class),
                    wrapper.getEnumValue("old_parent_type", NodeType.class),
                    wrapper.getEnumValue("old_parent_of_parent_type", NodeType.class),
                    wrapper.getStringValue("label"),
                    wrapper.getStringValue("old_label"));
            result.put(value, wrapper.getIntValue("count"));
        }
        return result;
    }

    @Override
    public void clean() {
        index.clean();
    }
}
