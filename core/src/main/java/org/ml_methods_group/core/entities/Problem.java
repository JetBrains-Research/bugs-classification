package org.ml_methods_group.core.entities;

import org.ml_methods_group.core.database.annotations.BinaryFormat;
import org.ml_methods_group.core.database.annotations.DataClass;
import org.ml_methods_group.core.database.annotations.DataField;

@DataClass(defaultStorageName = "problems")
public class Problem {
    @DataField
    private final int problemId;

    @BinaryFormat
    @DataField
    private final String text;

    public Problem(int problemId, String text) {
        this.problemId = problemId;
        this.text = text;
    }
}
