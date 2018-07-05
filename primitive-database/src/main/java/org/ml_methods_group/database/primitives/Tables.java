package org.ml_methods_group.database.primitives;

public final class Tables {
    public static final TableHeader CODES_HEADER = new TableHeader("codes",
            new Column("session_id", Type.INTEGER),
            new Column("verdict", Type.INTEGER),
            new Column("code", Type.BYTEA),
            new Column("problem_id", Type.INTEGER));

//    public static final TableHeader problems_header = new TableHeader("problems",
//            new Column("id", Type.INTEGER, true),
//            new Column("description", Type.BYTEA));

//    public static final TableHeader tags_header = new TableHeader("tags",
//            new Column("session_id", Type.TEXT),
//            new Column("tag", Type.TEXT));

    public static final TableHeader DIFF_HEADER = new TableHeader("diff",
            new Column("session_id", Type.INTEGER),
            new Column("action_type", Type.TEXT),
            new Column("node_type", Type.TEXT),
            new Column("parent_type", Type.TEXT),
            new Column("parent_of_parent_type", Type.TEXT),
            new Column("label", Type.TEXT),
            new Column("old_parent_type", Type.TEXT),
            new Column("old_parent_of_parent_type", Type.TEXT),
            new Column("old_label", Type.TEXT));

    public static final TableHeader CHANGE_CODE_INDEX_HEADER = new TableHeader("code_index",
            new Column("code", Type.BIGINT),
            new Column("encoding_type", Type.INTEGER),
            new Column("count", Type.INTEGER),
            new Column("change_type", Type.TEXT),
            new Column("node_type", Type.TEXT),
            new Column("parent_type", Type.TEXT),
            new Column("parent_of_parent_type", Type.TEXT),
            new Column("old_parent_type", Type.TEXT),
            new Column("old_parent_of_parent_type", Type.TEXT),
            new Column("label", Type.TEXT),
            new Column("old_label", Type.TEXT));

    public static final TableHeader LABEL_INDEX_HEADER = new TableHeader("label_index",
            new Column("label", Type.TEXT),
            new Column("id", Type.INTEGER),
            new Column("count", Type.INTEGER));
}
