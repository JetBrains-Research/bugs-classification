package org.ml_methods_group.core.basic.markers;

import org.ml_methods_group.core.ClusterMarker;
import org.ml_methods_group.core.entities.Solution;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class ManualMarker implements ClusterMarker<Solution, String> {

    private Scanner scanner = new Scanner(System.in);

    @Override
    public String mark(List<Solution> cluster) {
        final List<Solution> buffer = new ArrayList<>(cluster);
        Collections.shuffle(buffer);
        System.out.println("Cluster: (size: " + buffer.size() + ")");
        for (Solution solution : buffer.subList(0, Math.min(buffer.size(), 5))) {
            System.out.println("Session id: " + solution.getSessionId());
            System.out.println(solution.getCode());
            System.out.println();
        }
        System.out.print("Your mark:");
        return scanner.next();
    }
}
