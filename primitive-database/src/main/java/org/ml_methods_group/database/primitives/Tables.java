package org.ml_methods_group.database.primitives;

public final class Tables {
    public static final TableHeader codes_header = new TableHeader("codes",
            new Column("id", Type.TEXT, true),
            new Column("code", Type.BYTEA),
            new Column("problem", Type.INTEGER));

//    public static final TableHeader problems_header = new TableHeader("problems",
//            new Column("id", Type.INTEGER, true),
//            new Column("description", Type.BYTEA));

//    public static final TableHeader tags_header = new TableHeader("tags",
//            new Column("session_id", Type.TEXT),
//            new Column("tag", Type.TEXT));

    public static final TableHeader diff_header = new TableHeader("diff",
            new Column("session_id", Type.INTEGER),
            new Column("action_type", Type.INTEGER),
            new Column("node_type", Type.INTEGER),
            new Column("parent_type", Type.INTEGER),
            new Column("parent_of_parent_type", Type.INTEGER),
            new Column("label", Type.TEXT),
            new Column("old_parent", Type.INTEGER),
            new Column("old_parent_of_parent", Type.INTEGER),
            new Column("old_label", Type.TEXT));
}
