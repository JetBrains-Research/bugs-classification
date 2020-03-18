package org.ml_methods_group.cache;

import org.ml_methods_group.common.Database;
import org.ml_methods_group.common.Repository;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class HashDatabase implements Database {

    private final Path pathToDirectory;
    private final List<Repository> repositories = new ArrayList<>();

    public HashDatabase(Path pathToDirectory) {
        this.pathToDirectory = pathToDirectory;
    }

    @Override
    public <K, V> Repository<K, V> repositoryForName(String name,
                                                     Class<K> keyClass, Class<V> valueClass) throws Exception {
        final String fullName = keyClass.getSimpleName() + "2" + valueClass.getSimpleName() + "#" + name + ".cache";
        final var repository = new HashRepository<>(pathToDirectory.resolve(fullName), keyClass, valueClass);
        repositories.add(repository);
        return repository;
    }

    @Override
    public void close() throws Exception {
        final List<Exception> exceptions = new ArrayList<>();
        for (var repository : repositories) {
            try {
                repository.close();
            } catch (Exception e) {
                exceptions.add(e);
            }
        }
        if (!exceptions.isEmpty()) {
            final Exception exception = new Exception("Exceptions during database closing");
            exceptions.forEach(exception::addSuppressed);
            throw exception;
        }
    }
}
