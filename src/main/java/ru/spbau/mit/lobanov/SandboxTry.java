package ru.spbau.mit.lobanov;

import com.github.gumtreediff.actions.ActionGenerator;
import com.github.gumtreediff.gen.jdt.JdtTreeGenerator;
import com.github.gumtreediff.matchers.Matcher;
import com.github.gumtreediff.matchers.Matchers;
import com.github.gumtreediff.tree.ITree;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.sparse.SparseMLData;
import org.encog.neural.data.NeuralDataSet;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.simple.EncogUtility;
import ru.spbau.mit.lobanov.changes.ChangeUtils;
import ru.spbau.mit.lobanov.database.Database;
import ru.spbau.mit.lobanov.database.Table;
import ru.spbau.mit.lobanov.database.Tables;
import ru.spbau.mit.lobanov.preparation.DiffIndexer;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.sql.SQLException;
import java.util.*;

import static ru.spbau.mit.lobanov.preparation.DiffBuilder.insertDiffs;

public class SandboxTry {
    public static double XOR_INPUT[][] = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}};

    public static double XOR_IDEAL[][] = {{0.0}, {1.0}, {1.0}, {0.0}};

    public static void main(final String args[]) throws SQLException, IOException {
//
//        BasicNetwork network = EncogUtility.simpleFeedForward(2, 10, 10, 1,
//                false);
//        network.reset();
//
//        // Remove a few connections (does not really matter which, the network
//        // will train around them).
////        network.enableConnection(0, 0, 0, false);
////        network.enableConnection(0, 1, 3, false);
////        network.enableConnection(1, 1, 1, false);
////        network.enableConnection(1, 4, 4, false);
//        final List<MLDataPair> pairs = new ArrayList<>();
//        pairs.add(new BasicMLDataPair(new SparseMLData(new double[0], new int[0]), new BasicMLData(new double[]{0})));
//        pairs.add(new BasicMLDataPair(new SparseMLData(new double[]{1}, new int[]{0}), new BasicMLData(new double[]{1})));
//        pairs.add(new BasicMLDataPair(new SparseMLData(new double[]{1}, new int[]{1}), new BasicMLData(new double[]{1})));
//        pairs.add(new BasicMLDataPair(new SparseMLData(new double[]{1, 1}, new int[]{0, 1}), new BasicMLData(new double[]{0})));
//
//        NeuralDataSet trainingSet = new SparseMLDataSet(pairs);
//
//        ResilientPropagation rp = new ResilientPropagation(network, trainingSet);
//
//
//        System.out.println("Final output:");
//        EncogUtility.evaluate(network, trainingSet);
//
//        System.out
//                .println("Training should leave hidden neuron weights at zero.");
//        System.out.println("First removed neuron weight:" + network.getWeight(0, 0, 0) );
//        System.out.println("Second removed neuron weight:" + network.getWeight(0, 1, 3) );
//        System.out.println("Third removed neuron weight:" + network.getWeight(1, 1, 1) );
//        System.out.println("Fourth removed neuron weight:" + network.getWeight(1, 4, 4) );



//        try (Database database = new Database()) {
//            final Table codes = database.getTable(Tables.codes_header);
//            final String before = codes.findFirst("21037_0").getStringValue("code");
//            final String after = codes.findFirst("21037_1").getStringValue("code");
//            System.out.println(before);
//            System.out.println();
//            System.out.println(after);
//            System.out.println();
//            final ITree treeAfter = new JdtTreeGenerator().generateFromString(after).getRoot();
//            final ITree treeBefore = new JdtTreeGenerator().generateFromString(before).getRoot();
//            final Matcher matcher = Matchers.getInstance().getMatcher(treeBefore, treeAfter);
//            matcher.match();
//            while (true) {
//                final ActionGenerator g = new ActionGenerator(treeBefore, treeAfter, matcher.getMappings());
//                int s = g.generate().size();
////                if (s != 417) {
//                System.out.println(s);
////                }
//            }
//        }
        Database database = new Database();
        final Map<Long, Integer> index = DiffIndexer.getIndex(58088, database);
        FileUtils.writeDictionary("index2.txt", index);
    }
}
