#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

std::vector<std::vector>> 

struct HashMap {

    // Upcxx data structures 
    std::vector< upcxx::global_ptr<int> > used; // used array global pointer
    std::vector< upcxx::global_ptr<kmer_pair> > data; // data array global pointer
    size_t size_of_chunks; // segmentation size for each thread
    upcxx::atomic_domain<int32_t>* ad; // atomic domain class

    size_t my_size;

    size_t size() const noexcept;

    HashMap(size_t size);
    ~HashMap();

    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

HashMap::~HashMap() {
    delete ad;
}

HashMap::HashMap(size_t size) {
    my_size = size;
    
    // calculate starting index for each rank to store data
    size_of_chunks = (my_size + upc::rank_n() - 1) / upc::rank_n();
    data.resize(upc::rank_n());
    used.resize(upc::rank_n());
    // initialize data and used arrays
    size_t start_idx = upc::rank_me() * size_of_chunks;
    size_t end_idx = std::min(start_idx + size_of_chunks, my_size);
    data[upc::rank_me()] = upcxx::new_array<kmer_pair>(end_idx - start_idx);
    used[upc::rank_me()] = upcxx::new_array<int>(end_idx - start_idx);
    // broadcast arrays to all ranks
    for (int i = 0; i < upc::rank_n(); i++) {
        data[i] = upcxx::broadcast(data[i], i).wait();
        used[i] = upcxx::broadcast(used[i], i).wait();
    }
    // local pointer to the used array
    int* used_local = used[upc::rank_me].local();
    size_t start_local = upc::rank_me() * size_of_chunks;
    size_t end_local = std::min(start_local + size_of_chunks, my_size);
    // fill the local used array with zeros
    upcxx::fill_n(used_local, end_local - start_local, 0).wait(); 
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
}

// TODO: haven`t implemented the find function yet
bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true;
            }
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::slot_used(uint64_t slot) { 
    int rank = slot / split;
    int index = slot % split;
    return upcxx::rget(used[rank] + index).wait() != 0;
 }

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    int rank = slot / split;
    int index = slot % split;
    return upcxx::rput(kmer, data[rank] + index).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) { 
    int rank = slot / split;
    int index = slot % split;
    return upcxx::rget(data[rank] + index).wait(); 
}

bool HashMap::request_slot(uint64_t slot) {
    // if (used[slot] != 0) {
    //     return false;
    // } else {
    //     used[slot] = 1;
    //     return true;
    // }
    int rank = slot / split;
    int index = slot % split;
    // Atomically check and update the value of used[rank] + index
    int current_value = ad->atomic_exchange(&used[rank] + index, 1);

    // If the previous value was not 0, the slot was already used
    if (current_value != 0) {
        return false;
    } else {
        return true;
    }
}

size_t HashMap::size() const noexcept { return my_size; }
