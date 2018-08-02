package org.ml_methods_group;

import org.ml_methods_group.core.entities.CachedDecision;
import org.ml_methods_group.core.entities.CachedDistance;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.preparation.JavaCodeValidator;
import org.ml_methods_group.core.preparation.PreparationUtils;
import org.ml_methods_group.database.Database;
import org.ml_methods_group.database.SQLRepository;

import java.io.IOException;

public class InitScript {

    public static final int PROBLEM = 58088;
    //    public static final int PROBLEM = 47329;
    //    public static final int PROBLEM = 47347;

    public static void main(String[] args) throws IOException {
        final Database database = new Database();
        clear(database);
        PreparationUtils.parse(
                InitScript.class.getResourceAsStream("/dataset.csv"),
                new JavaCodeValidator(),
                x -> x == PROBLEM,
                new SQLRepository<>(Solution.class, database));
    }

    private static void clear(Database database) {
        new SQLRepository<>(CachedDecision.class, database).clear();
        new SQLRepository<>(CachedDistance.class, database).clear();
        new SQLRepository<>(Solution.class, database).clear();
    }
}
