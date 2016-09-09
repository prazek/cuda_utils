#pragma once
#include <vector>
#include <limits>
#include <cuda_runtime.h>
#include <helper_cuda.h>

template <typename T>
class device_vector;

template <typename T>
class device_ptr {
  T *ptr_ = nullptr;

private:
  device_ptr(T *ptr_d) : ptr_(ptr_d) {}
  template <typename P>
  friend device_ptr<P> make_device_ptr_from_raw_ptr(P *ptr_d);
  friend class device_vector<T>;
public:
  device_ptr() = default;
  const T *dev() const {
    return ptr_;
  }
  T* dev() {
    return ptr_;
  }

  void free() {
    checkCudaErrors(cudaFree(ptr_));
  }
};

template <typename T>
class device_vector : public device_ptr<T> {
  int size_ = 0;

private:
  device_vector(T* ptr_d, int size) : device_ptr<T>(ptr_d), size_(size) {}
  template <typename P>
  friend device_vector<P> make_device_vector_from_raw_ptr(P *ptr_d, int size);
public:
  int size() const { return size_;}
};

template <typename T>
class pinned_allocator {
public:
  // type definitions
  typedef T        value_type;
  typedef T*       pointer;
  typedef const T* const_pointer;
  typedef T&       reference;
  typedef const T& const_reference;
  typedef std::size_t    size_type;
  typedef std::ptrdiff_t difference_type;

  // rebind allocator to type U
  template <class U>
  struct rebind {
    typedef pinned_allocator<U> other;
  };

  // return address of values
  pointer address (reference value) const {
    return &value;
  }
  const_pointer address (const_reference value) const {
    return &value;
  }

  /* constructors and destructor
   * - nothing to do because the allocator has no state
   */
  pinned_allocator() throw() {
  }
  pinned_allocator(const pinned_allocator&) throw() {
  }
  template <class U>
  pinned_allocator (const pinned_allocator<U>&) throw() {
  }
  ~pinned_allocator() throw() {
  }

  // return maximum number of elements that can be allocated
  size_type max_size () const throw() {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

  // allocate but don't initialize num elements of type T
  pointer allocate (size_type num, const void* = 0) {
    pointer ret;
    checkCudaErrors(cudaMallocHost((void**)&ret, num * sizeof(T)));
    return ret;
  }

  // initialize elements of allocated storage p with value value
  void construct (pointer p, const T& value) {
    // initialize memory with placement new
    new((void*)p)T(value);
  }

  // destroy elements of initialized storage p
  void destroy (pointer p) {
    // destroy objects by calling their destructor
    p->~T();
  }

  // deallocate storage p of deleted elements
  void deallocate (pointer p, size_type num) {
    checkCudaErrors(cudaFreeHost((void*)p));
  }
};

// return that all specializations of this allocator are interchangeable
template <class T1, class T2>
bool operator== (const pinned_allocator<T1>&,
                 const pinned_allocator<T2>&) throw() {
  return true;
}
template <class T1, class T2>
bool operator!= (const pinned_allocator<T1>&,
                 const pinned_allocator<T2>&) throw() {
  return false;
}



/// Makes device pointer from other device pointer.
/// Long name is intentional, becuase this function should not be normally used,
/// because it can easily leed to device/host pointer bugs.
template <typename T>
device_ptr<T> make_device_ptr_from_raw_ptr(T* ptr_d) {
  return device_ptr<T>(ptr_d);
}

/// Makes device vector from raw device pointer and size.
/// Long name is intentional, becuase this function should not be normally used,
/// because it can easily leed to device/host pointer bugs.
template <typename T>
device_vector<T> make_device_vector_from_raw_ptr(T* ptr_d, int size) {
  return device_vector<T>(ptr_d, size);
}

template <typename T>
device_vector<T> make_device_vector(int size) {
  T* array_d;
  checkCudaErrors(cudaMalloc((void**)&array_d, sizeof(T) * size));
  return make_device_vector_from_raw_ptr(array_d, size);
}

template <typename T, typename AllocType>
device_vector<T> make_device_vector(const std::vector<T, AllocType>& vec) {
  device_vector<T> ptr = make_device_vector<T>(vec.size());
  checkCudaErrors(cudaMemcpy(ptr.dev(), vec.data(), sizeof(T) * vec.size(),
                             cudaMemcpyHostToDevice));
  return ptr;
}

template <typename T, typename AllocType>
void copy_to_vector(device_ptr<T> ptr, std::vector<T, AllocType> &vec, int size) {
  if (vec.size() < size)
    vec.resize(size);
  checkCudaErrors(cudaMemcpy(vec.data(), ptr.dev(),
                             sizeof(T) * size,
                             cudaMemcpyDeviceToHost));
}

template <typename T, typename AllocType>
void copy_to_vector(device_vector<T> dev_vec, std::vector<T, AllocType> &vec) {
  copy_to_vector(dev_vec, vec, dev_vec.size());
}


template <typename T, typename AllocType=std::allocator<T>>
std::vector<T, AllocType> make_vector(device_ptr<T> ptr, int size) {
  std::vector<T, AllocType> result(size);
  copy_to_vector(ptr, result, size);
  return result;
}

template <typename T, typename AllocType=std::allocator<T>>
std::vector<T, AllocType> make_vector(device_vector<T> dev_vec) {

  return make_vector<T, AllocType>(dev_vec, dev_vec.size());
}