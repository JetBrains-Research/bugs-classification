package ru.spbau.mit.lobanov.database;

public final class Tables {
    public static final TableHeader codes_header = new TableHeader("codes",
            new Column("id", Type.TEXT, true),
            new Column("code", Type.BYTEA),
            new Column("problem", Type.INTEGER));

    public static final TableHeader problems_header = new TableHeader("problems",
            new Column("id", Type.INTEGER, true),
            new Column("description", Type.BYTEA));

    public static final TableHeader tags_header = new TableHeader("tags",
            new Column("session_id", Type.TEXT),
            new Column("tag", Type.TEXT));

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

    public static final TableHeader labels_header = new TableHeader("labels",
            new Column("label", Type.TEXT, true),
            new Column("id", Type.INTEGER),
            new Column("count", Type.INTEGER));

    public static final TableHeader diff_types_header = new TableHeader("diff_types",
            new Column("id", Type.BIGINT, true),
            new Column("action_type", Type.INTEGER),
            new Column("node_type", Type.INTEGER),
            new Column("parent_type", Type.INTEGER),
            new Column("parent_of_parent_type", Type.INTEGER),
            new Column("label", Type.INTEGER),
            new Column("old_parent", Type.INTEGER),
            new Column("old_parent_of_parent", Type.INTEGER),
            new Column("old_label", Type.INTEGER),
            new Column("count", Type.INTEGER));

    public static final TableHeader train_pairs_header = new TableHeader("train_pairs",
            new Column("first", Type.INTEGER),
            new Column("second", Type.INTEGER),
            new Column("is_similar", Type.INTEGER));
}
