package org.ml_methods_group.clustering.clusterers;

import org.ml_methods_group.common.Cluster;
import org.ml_methods_group.common.Clusterer;
import org.ml_methods_group.common.Clusters;
import org.ml_methods_group.common.DistanceFunction;
import org.ml_methods_group.common.parallel.ParallelContext;
import org.ml_methods_group.common.parallel.ParallelUtils;

import java.util.*;
import java.util.stream.Collectors;

public class HAC<T> implements Clusterer<T> {

    protected final SortedSet<Triple> heap = new TreeSet<>();
    protected final Map<Long, Triple> triples = new HashMap<>();
    protected final Set<Community> communities = new HashSet<>();
    private final double distanceLimit;
    private int minClustersCount;
    private final DistanceFunction<T> metric;
    private int idGenerator = 0;

    public HAC(double distanceLimit, int minClustersCount, DistanceFunction<T> metric) {
        this.distanceLimit = distanceLimit;
        this.minClustersCount = minClustersCount;
        this.metric = metric;
    }

    protected HAC(double distanceLimit, DistanceFunction<T> metric) {
        this.distanceLimit = distanceLimit;
        this.metric = metric;
    }

    protected void init(List<T> values) {
        heap.clear();
        triples.clear();
        communities.clear();
        idGenerator = 0;
        values.stream()
                .map(this::singletonCommunity)
                .forEach(communities::add);
        final List<Community> communitiesAsList = new ArrayList<>(communities);
        Collections.shuffle(communitiesAsList);
        try (ParallelContext context = new ParallelContext()) {
            final List<Triple> toInsert = context.runParallel(communitiesAsList,
                    ArrayList::new,
                    this::findTriples,
                    ParallelUtils::combineLists);
            toInsert.forEach(this::insertTriple);
        }
    }

    private List<Triple> findTriples(Community community, List<Triple> accumulator) {
        final T representative = community.entities.get(0);
        for (Community another : communities) {
            if (another == community) {
                break;
            }
            final double distance = metric.distance(representative, another.entities.get(0), distanceLimit);
            if (distance < distanceLimit) {
                accumulator.add(new Triple(distance, community, another));
            }
        }
        return accumulator;
    }

    @Override
    public Clusters<T> buildClusters(List<T> values) {
        init(values);
        while (!heap.isEmpty() && communities.size() > minClustersCount) {
            final Triple minTriple = heap.first();
            invalidateTriple(minTriple);
            final Community first = minTriple.first;
            final Community second = minTriple.second;
            mergeCommunities(first, second);
        }
        final List<Cluster<T>> clusters = communities.stream()
                .map(c -> c.entities)
                .map(Cluster::new)
                .collect(Collectors.toList());
        return new Clusters<>(clusters);
    }

    protected void mergeCommunities(Community first, Community second) {
        final List<T> merged;
        if (first.entities.size() < second.entities.size()) {
            merged = second.entities;
            merged.addAll(first.entities);
        } else {
            merged = first.entities;
            merged.addAll(second.entities);
        }

        final Community newCommunity = new Community(merged);
        communities.remove(first);
        communities.remove(second);

        for (Community community : communities) {
            final long fromFirstID = getTripleID(first, community);
            final long fromSecondID = getTripleID(second, community);
            final Triple fromFirst = triples.get(fromFirstID);
            final Triple fromSecond = triples.get(fromSecondID);
            final double newDistance = Math.max(getDistance(fromFirst), getDistance(fromSecond));
            invalidateTriple(fromFirst);
            invalidateTriple(fromSecond);
            insertTripleIfNecessary(newDistance, newCommunity, community);
        }
        communities.add(newCommunity);
    }

    private double getDistance(Triple triple) {
        return triple == null ? Double.POSITIVE_INFINITY : triple.distance;
    }

    private long getTripleID(Community first, Community second) {
        if (second.id > first.id) {
            return getTripleID(second, first);
        }
        return first.id * 1_000_000_009L + second.id;
    }

    private void insertTriple(Triple triple) {
        triples.put(getTripleID(triple.first, triple.second), triple);
        heap.add(triple);
    }

    protected void insertTripleIfNecessary(double distance, Community first, Community second) {
        if (distance >= distanceLimit) {
            return;
        }
        final Triple triple = createTriple(distance, first, second);
        insertTriple(triple);
    }

    protected void invalidateTriple(Triple triple) {
        if (triple == null) {
            return;
        }
        final long tripleID = getTripleID(triple.first, triple.second);
        triples.remove(tripleID);
        heap.remove(triple);
        triple.release();
    }

    private Community singletonCommunity(T entity) {
        final List<T> singletonList = new ArrayList<>(1);
        singletonList.add(entity);
        return new Community(singletonList);
    }

    protected class Community implements Comparable<Community> {

        protected final List<T> entities;
        private final int id;

        Community(List<T> entities) {
            this.entities = entities;
            id = idGenerator++;
        }

        @Override
        public int compareTo(Community o) {
            return id - o.id;
        }

        @Override
        public int hashCode() {
            return id;
        }

        @Override
        public boolean equals(Object obj) {
            return obj.getClass() == Community.class && ((Community) obj).id == id;
        }
    }

    protected class Triple implements Comparable<Triple> {

        private double distance;
        protected Community first;
        protected Community second;

        Triple(double distance, Community first, Community second) {
            this.distance = distance;
            this.first = first;
            this.second = second;
        }

        void release() {
            triplesPoll.add(this);
        }

        @Override
        public int compareTo(Triple other) {
            if (other == this) {
                return 0;
            }
            if (distance != other.distance) {
                return Double.compare(distance, other.distance);
            }
            if (first != other.first) {
                return first.compareTo(other.first);
            }
            return second.compareTo(other.second);
        }
    }

    private final Queue<Triple> triplesPoll = new ArrayDeque<>();

    private Triple createTriple(double distance, Community first, Community second) {
        if (triplesPoll.isEmpty()) {
            return new Triple(distance, first, second);
        }
        final Triple triple = triplesPoll.poll();
        triple.distance = distance;
        triple.first = first;
        triple.second = second;
        return triple;
    }
}