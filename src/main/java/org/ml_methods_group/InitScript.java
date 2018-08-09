package org.ml_methods_group;

import org.ml_methods_group.core.database.Repository;
import org.ml_methods_group.core.entities.Solution;
import org.ml_methods_group.core.preparation.JavaCodeValidator;
import org.ml_methods_group.core.preparation.PreparationUtils;
import org.ml_methods_group.database.SQLDatabase;

import java.io.IOException;

public class InitScript {

//    public static final int PROBLEM = 58088; // find min max             ok
//        public static final int PROBLEM = 47329;                         ?
//        public static final int PROBLEM = 47347;                         ?
//        public static final int PROBLEM = 57810;                         ?
//        public static final int PROBLEM = 47333; // check if palindrom   ?
//        public static final int PROBLEM = 164397; // small dataset       ?
//        public static final int PROBLEM = 39716;// fibbonachi              ok                       s1
//        public static final int PROBLEM = 72875;// mapper                ок (not very interesting)
//        public static final int PROBLEM = 57792;// long code             clusters - ok, classification - :(
        public static final int PROBLEM = 55715;//    deserialization     ok (similar solutions)      s2

    public static void main(String[] args) throws IOException {
        final SQLDatabase database = new SQLDatabase();
        final Repository<Solution> solutions = database.getRepository(Solution.class);
        if (!solutions.find(solutions.conditionSupplier().is("problemid", PROBLEM)).isPresent()) {
            PreparationUtils.parse(
                    InitScript.class.getResourceAsStream("/dataset.csv"),
                    new JavaCodeValidator(),
                    x -> x == PROBLEM,
                    solutions);
        }
    }
}
