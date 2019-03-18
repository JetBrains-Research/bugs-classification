package org.ml_methods_group.cache;

import org.ml_methods_group.common.Repository;

import java.io.*;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Optional;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class HashRepository<K, V> implements Repository<K, V> {

    private final Path pathToStorage;
    private final HashMap<K, V> cache;
    private final ReadWriteLock lockManager = new ReentrantReadWriteLock();

    public HashRepository(Path pathToStorage, Class<K> keyClass, Class<V> valueClass) throws Exception {
        this.pathToStorage = pathToStorage;
        this.cache = loadMap(pathToStorage, keyClass, valueClass);
    }


    @Override
    public Optional<V> loadValue(K key) {
        final Lock lock = lockManager.readLock();
        lock.lock();
        final V value = cache.get(key);
        lock.unlock();
        return Optional.ofNullable(value);
    }

    @Override
    public void storeValue(K key, V value) {
        final Lock lock = lockManager.writeLock();
        lock.lock();
        cache.put(key, value);
        if (cache.size() % 10 == 0) {
            System.out.println("Size: " + cache.size());
        }
        lock.unlock();
    }

    @Override
    public void close() throws IOException {
        flush();
    }

    private void flush() throws IOException {
        final Lock lock = lockManager.readLock();
        lock.lock();
        final Path directory = pathToStorage.getParent();
        if (directory != null && !directory.toFile().exists() && !directory.toFile().mkdirs()) {
            throw new IOException("Failed to create parent directories: " + directory.toString());
        }
        try (FileOutputStream fileOutputStream = new FileOutputStream(pathToStorage.toFile());
             ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
            final var entrySet = cache.entrySet();
            objectOutputStream.writeInt(entrySet.size());
            for (var entry : entrySet) {
                objectOutputStream.writeObject(entry.getKey());
                objectOutputStream.writeObject(entry.getValue());
            }
        } finally {
            lock.unlock();
        }
    }

    private static <K, V> HashMap<K, V> loadMap(Path path, Class<K> keyClass, Class<V> valueClass) throws Exception {
        try (FileInputStream fileInputStream = new FileInputStream(path.toFile());
             ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            final HashMap<K, V> result = new HashMap<>();
            final int size = objectInputStream.readInt();
            for (int i = 0; i < size; i++) {
                final K key = keyClass.cast(objectInputStream.readObject());
                final V value = valueClass.cast(objectInputStream.readObject());
                result.put(key, value);
            }
            return result;
        } catch (FileNotFoundException e) {
            return new HashMap<>();
        }
    }
}
