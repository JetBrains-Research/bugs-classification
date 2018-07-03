import org.ml_methods_group.classification.KNearestNeighbors;
import org.ml_methods_group.core.Utils;
import org.ml_methods_group.core.vectorization.VectorTemplate;
import org.ml_methods_group.database.BasicIndexStorage;
import org.ml_methods_group.database.proxy.ProxySolutionDatabase;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws SQLException, IOException {
        ProxySolutionDatabase database = new ProxySolutionDatabase();
        BasicIndexStorage storage = new BasicIndexStorage();
//        Utils.loadDatabase(Main.class.getResourceAsStream("/dataset.csv"), database, 55673);
        final String ids = Utils.indexLabels(database, storage, 50, 100000);
        final Map<String, Long> dictionary = storage.loadIndex(ids);
        Utils.generateTemplate(database, storage, Utils.defaultStrategies(dictionary),
                "index1", VectorTemplate.BasePostprocessors.RELATIVE, 10, Integer.MAX_VALUE);
        KNearestNeighbors t;
    }
}
