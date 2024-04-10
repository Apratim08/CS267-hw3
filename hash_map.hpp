#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <math.h>


struct HashMap {
    using GlobalPtrsToKmer = std::vector<upcxx::global_ptr<kmer_pair>>;
    using GlobalPtrsToInt = std::vector<upcxx::global_ptr<int>>;
    
    GlobalPtrsToKmer data;
    GlobalPtrsToInt used;
	
    size_t my_size;
	
    size_t globalHashTableSize;
	
	size_t size() const noexcept;
	
    size_t totalGlobalSize() const noexcept;

    HashMap(size_t size);

    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);

private: 
    upcxx::atomic_domain<int> atomicDomain;
};

HashMap::HashMap(size_t size)
    : globalHashTableSize(size),
	
      atomicDomain(upcxx::atomic_domain<int>({upcxx::atomic_op::fetch_add})) 
{	
	double tempSize = 
	static_cast<double>(globalHashTableSize) / static_cast<double>(upcxx::rank_n());
	
    my_size = std::ceil(tempSize);
	
    used.resize(upcxx::rank_n());
    data.resize(upcxx::rank_n());
	
    for (int i = 0; i < upcxx::rank_n(); ++i) {
        if (i == upcxx::rank_me()) {
            used[i] = upcxx::new_array<int>(my_size);
            data[i] = upcxx::new_array<kmer_pair>(my_size);
        }
        used[i] = upcxx::broadcast(used[i], i).wait();
        data[i] = upcxx::broadcast(data[i], i).wait();       
    }
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe) % totalGlobalSize();
        success = request_slot(slot); 
        if (success) {
            write_slot(slot, kmer);
        }
        ++probe;
    } while (!success && probe < totalGlobalSize());
    return success;
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0; 
    bool success = false;
    do {
        uint64_t slot = (hash + probe) % totalGlobalSize(); 
        if (slot_used(slot)) {
            val_kmer = read_slot(slot);
            if (val_kmer.kmer == key_kmer) {
                success = true; 
            }
        }
        ++probe; 
    } while (!success && probe < totalGlobalSize()); 
    return success;
}

bool HashMap::slot_used(uint64_t slot) { 
    auto ptrToUsed = used[slot / size()] + slot % size();
    return upcxx::rget(ptrToUsed).wait() != 0; 
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    auto ptrToData = data[slot / size()] + slot % size();
    upcxx::rput(kmer, ptrToData).wait(); 
}

kmer_pair HashMap::read_slot(uint64_t slot) { 
    auto ptrToData = data[slot / size()] + slot % size();
    return upcxx::rget(ptrToData).wait(); 
}

bool HashMap::request_slot(uint64_t slot) { 
    auto ptrToUsed = used[slot / size()] + slot % size();
    return atomicDomain.fetch_add(ptrToUsed, 1, std::memory_order_relaxed).wait() == 0; 
}


size_t HashMap::size() const noexcept { return my_size; }

size_t HashMap::totalGlobalSize() const noexcept { return globalHashTableSize; }
