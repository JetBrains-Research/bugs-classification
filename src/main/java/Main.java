import org.ml_methods_group.core.IndexDatabase;
import org.ml_methods_group.core.changes.AtomicChange;
import org.ml_methods_group.core.vectorization.EncodingStrategy;
import org.ml_methods_group.database.BasicIndexDatabase;
import org.ml_methods_group.database.primitives.Database;
import org.ml_methods_group.database.proxy.ProxyDatabase;
import org.ml_methods_group.core.vectorization.ChangeEncodingStrategy;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Collections;

public class Main {
    public static void main(String[] args) throws SQLException, IOException {
        Database database = new Database();
        ProxyDatabase proxy = new ProxyDatabase(database);
        IndexDatabase index = new BasicIndexDatabase(database);
//        CSVParser.parse(new File("dataset.csv"), proxy);
//        proxy.findByProblem(55673).stream().forEachOrdered(SolutionDiff::getChanges);
        EncodingStrategy<AtomicChange> strategy = new ChangeEncodingStrategy(Collections.emptyMap(), 1,
                ChangeEncodingStrategy.ChangeAttribute.values());

        database.close();
    }
}
