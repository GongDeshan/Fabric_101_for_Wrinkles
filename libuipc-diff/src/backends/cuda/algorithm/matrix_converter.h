#pragma once
#include <type_define.h>
#include <muda/buffer/device_buffer.h>
#include <muda/ext/linear_system/matrix_format_converter.h>

namespace uipc::backend::cuda
{
template <typename T, int M, int N = M>
class MatrixConverter
{
    using BlockMatrix   = muda::DeviceTripletMatrix<T, M, N>::ValueT;
    using SegmentVector = muda::DeviceDoubletVector<T, M>::ValueT;

    Float m_reserve_ratio = 1.5;

    muda::DeviceBuffer<int> col_counts_per_row;
    muda::DeviceBuffer<int> unique_indices;
    muda::DeviceBuffer<int> unique_counts;
    muda::DeviceVar<int>    count;

    muda::DeviceBuffer<int> sort_index_input;
    muda::DeviceBuffer<int> sort_index;

    muda::DeviceBuffer<int> offsets;

    muda::DeviceBuffer<int2> ij_pairs;
    muda::DeviceBuffer<int2> unique_ij_pairs;

    muda::DeviceBuffer<uint64_t> ij_hash_input;
    muda::DeviceBuffer<uint64_t> ij_hash;

    muda::DeviceBuffer<BlockMatrix> blocks_sorted;
    muda::DeviceBuffer<BlockMatrix> diag_blocks;


    muda::DeviceBuffer<int>           indices_sorted;
    muda::DeviceBuffer<SegmentVector> segments_sorted;


    muda::DeviceBuffer<int> sorted_partition_input;
    muda::DeviceBuffer<int> sorted_partition_output;

  public:
    void  reserve_ratio(Float ratio) { m_reserve_ratio = ratio; }
    Float reserve_ratio() const { return m_reserve_ratio; }


    // Triplet -> BCOO
    void convert(const muda::DeviceTripletMatrix<T, M, N>& from,
                 muda::DeviceBCOOMatrix<T, M, N>&          to);

    void _radix_sort_indices_and_blocks(const muda::DeviceTripletMatrix<T, M, N>& from,
                                        muda::DeviceBCOOMatrix<T, M, N>& to);

    void _radix_sort_indices_and_blocks(muda::DeviceBCOOMatrix<T, M, N>& to);

    void _make_unique_indices(const muda::DeviceTripletMatrix<T, M, N>& from,
                              muda::DeviceBCOOMatrix<T, M, N>&          to);

    void _make_unique_block_warp_reduction(const muda::DeviceTripletMatrix<T, M, N>& from,
                                           muda::DeviceBCOOMatrix<T, M, N>& to);

    // BCOO -> BSR
    void convert(const muda::DeviceBCOOMatrix<T, M, N>& from,
                 muda::DeviceBSRMatrix<T, N>&           to);

    void _calculate_block_offsets(const muda::DeviceBCOOMatrix<T, M, N>& from,
                                  muda::DeviceBSRMatrix<T, N>&           to);


    // Doublet -> BCOO
    void convert(const muda::DeviceDoubletVector<T, M>& from,
                 muda::DeviceBCOOVector<T, M>&          to);

    void _radix_sort_indices_and_segments(const muda::DeviceDoubletVector<T, M>& from,
                                          muda::DeviceBCOOVector<T, M>& to);

    void _make_unique_indices(const muda::DeviceDoubletVector<T, M>& from,
                              muda::DeviceBCOOVector<T, M>&          to);

    void _make_unique_segment_warp_reduction(const muda::DeviceDoubletVector<T, M>& from,
                                             muda::DeviceBCOOVector<T, M>& to);


    template <typename U>
    void loose_resize(muda::DeviceBuffer<U>& buf, size_t new_size)
    {
        if(buf.capacity() < new_size)
            buf.reserve(new_size * m_reserve_ratio);
        buf.resize(new_size);
    }

    void ge2sym(muda::DeviceBCOOMatrix<T, M, N>& to)
        requires(M == N);

    void sym2ge(const muda::DeviceBCOOMatrix<T, M, N>& from,
                muda::DeviceBCOOMatrix<T, M, N>&       to)
        requires(M == N);
};
}  // namespace uipc::backend::cuda

#include "details/matrix_converter.inl"