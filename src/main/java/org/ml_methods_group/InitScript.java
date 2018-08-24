package org.ml_methods_group;

import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.preparation.JavaCodeValidator;
import org.ml_methods_group.core.preparation.PreparationUtils;
import org.ml_methods_group.database.SQLDatabase;

import java.io.FileInputStream;
import java.io.IOException;

public class InitScript {

    public static void main(String[] args) throws IOException {
        final String filename = args[0];
        final SQLDatabase database = new SQLDatabase();
        final Repository<Solution> solutions = database.getRepository(Solution.class);
        PreparationUtils.parse(
                new FileInputStream(filename),
                new JavaCodeValidator(),
                x -> true,
                solutions);
    }
}
