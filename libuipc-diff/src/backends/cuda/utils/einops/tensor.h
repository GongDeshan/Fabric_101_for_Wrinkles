#ifdef _MSC_VER
#pragma once
#include <muda/muda_def.h>
#include <muda/tools/debug_log.h>
#include <Eigen/Core>

namespace muda::einops
{
template <typename T, int N>
class IndexArray
{
  public:
    constexpr MUDA_GENERIC IndexArray() noexcept = default;

    template <typename... Args>
    MUDA_GENERIC IndexArray(Args... args) noexcept
        : m_data{args...}
    {
        static_assert(sizeof...(args) == N, "Size mismatch");
    }

    MUDA_GENERIC const T& operator[](int i) const noexcept
    {
        MUDA_ASSERT(i >= 0 && i < N, "Index out of range: %d, %d", i, N);
        return m_data[i];
    }

    MUDA_GENERIC T& operator[](int i) noexcept
    {
        MUDA_ASSERT(i >= 0 && i < N, "Index out of range: %d, %d", i, N);
        return m_data[i];
    }

    MUDA_GENERIC const T* data() const noexcept
    {
        return N == 0 ? nullptr : m_data;
    }

    MUDA_GENERIC T* data() noexcept { return N == 0 ? nullptr : m_data; }

    MUDA_GENERIC static IndexArray iota() noexcept
    {
        IndexArray result;
        if constexpr(N > 0)
        {
            for(int i = 0; i < N; ++i)
            {
                result[i] = i;
            }
        }
        return result;
    }

  private:
    constexpr static size_t ArraySize = (N == 0) ? 1 : N;

    T m_data[ArraySize] = {};
};

template <typename T>
using NullArray = IndexArray<T, 0>;

constexpr NullArray<int> null;

template <typename T, int... Shape>
class NDArrayView
{
  public:
    constexpr static int Dim = sizeof...(Shape);
    using ValueType          = T;
    using ShapeSequence      = std::integer_sequence<int, Shape...>;

    MUDA_GENERIC NDArrayView(T* data) noexcept
        : m_data(data)
        , m_shape{Shape...}
    {
        // Using Column-Major Order
        if constexpr(Dim > 0)
        {
            m_strides[0] = 1;
            for(int i = 1; i < Dim; ++i)
            {
                m_strides[i] = m_strides[i - 1] * m_shape[i - 1];
            }
        }
    }

    MUDA_GENERIC const T& get(const IndexArray<int, Dim>& index) const noexcept
    {
        return m_data[get_index(index)];
    }

    MUDA_GENERIC void set(T value, const IndexArray<int, Dim>& index) noexcept
    {
        m_data[get_index(index)] = value;
    }

    MUDA_GENERIC int shape(int i) const noexcept
    {
        if constexpr(Dim == 0)
            return 1;
        else
            return m_shape[i];
    }

  private:
    MUDA_GENERIC int get_index(const IndexArray<int, Dim>& index_array) const noexcept
    {
        if constexpr(Dim > 0)
        {
            int result = 0;

            for(int i = 0; i < Dim; ++i)
            {
                auto I = i;

                MUDA_ASSERT(index_array[I] >= 0 && index_array[I] < m_shape[i],
                            "Index out of range: %d, %d",
                            index_array[I],
                            m_shape[i]);

                result += index_array[I] * m_strides[i];
            }

            // printf("result: %d\n", result);

            return result;
        }
        else
        {
            return 0;
        }
    }

    T* m_data = nullptr;

    IndexArray<int, Dim> m_shape;
    IndexArray<int, Dim> m_strides;
};

template <typename T, int M, int N>
MUDA_GENERIC NDArrayView<T, M, N> array_view(Eigen::Matrix<T, M, N>& mat)
{
    return NDArrayView<T, M, N>(mat.data());
}

template <typename T, int... Shape>
class NDArray
{
  public:
    constexpr static int Dim = sizeof...(Shape);
    using ShapeSequence      = std::integer_sequence<int, Shape...>;
    using ValueType          = T;

    MUDA_GENERIC NDArray() noexcept
        : m_shape{Shape...}
    {
        // Using Column-Major Order
        if constexpr(Dim > 0)
        {
            m_strides[0] = 1;
            for(int i = 1; i < Dim; ++i)
            {
                m_strides[i] = m_strides[i - 1] * m_shape[i - 1];
            }
        }
    }

    MUDA_GENERIC const T& get(const IndexArray<int, Dim>& index) const noexcept
    {
        return m_data[get_index(index)];
    }

    MUDA_GENERIC void set(T value, const IndexArray<int, Dim>& index) noexcept
    {
        m_data[get_index(index)] = value;
    }

    MUDA_GENERIC int shape(int i) const noexcept { return m_shape[i]; }

  private:
    MUDA_GENERIC int get_index(const IndexArray<int, Dim>& index_array) const noexcept
    {
        if constexpr(Dim > 0)
        {
            int result = 0;

            for(int i = 0; i < Dim; ++i)
            {
                auto I = i;  // reverse order

                MUDA_ASSERT(index_array[I] >= 0 && index_array[I] < m_shape[i],
                            "Index out of range: %d, %d",
                            index_array[I],
                            m_shape[i]);

                result += index_array[I] * m_strides[i];
            }

            return result;
        }
        else
        {
            return 0;
        }
    }

    T m_data[(Shape * ...)] = {};

    IndexArray<int, Dim> m_shape;
    IndexArray<int, Dim> m_strides;
};

template <typename T>
class NDArray<T>
{
  public:
    constexpr static int Dim = 0;
    using ValueType          = T;
    using ShapeSequence      = std::integer_sequence<int>;


    MUDA_GENERIC NDArray() noexcept = default;

    MUDA_GENERIC const T& get(const IndexArray<int, Dim>& index) const noexcept
    {
        return m_data;
    }

    MUDA_GENERIC void set(T value, const IndexArray<int, Dim>& index) noexcept
    {
        m_data = value;
    }

    MUDA_GENERIC int shape(int i) const noexcept { return 1; }

  private:
    T m_data = {};
};

template <typename T>
using Scalar = NDArray<T>;

template <typename T, typename U>
struct ndarray_maker;

template <typename T, int... Is>
struct ndarray_maker<T, std::integer_sequence<int, Is...>>
{
    using type = NDArray<T, Is...>;
};

template <typename T, typename U>
using ndarray_maker_t = typename ndarray_maker<T, U>::type;

template <int Index, typename T, T... Ints>
constexpr T get(std::integer_sequence<T, Ints...>)
{
    constexpr T arr[] = {Ints...};
    return arr[Index];
}

template <int Index, int Val, int... Ints, int... Is>
constexpr auto set_impl(std::integer_sequence<int, Ints...>, std::index_sequence<Is...>)
{
    return std::integer_sequence<int, (Is == Index ? Val : Ints)...>{};
}

template <int Index, int Val, int... Ints>
constexpr auto set(std::integer_sequence<int, Ints...> seq)
{
    return set_impl<Index, Val>(seq, std::make_index_sequence<sizeof...(Ints)>{});
}

template <int... Is, int... Js>
constexpr auto concat(std::integer_sequence<int, Is...>, std::integer_sequence<int, Js...>)
{
    return std::integer_sequence<int, Is..., Js...>{};
}

template <int Val, int... Is>
constexpr auto push_back(std::integer_sequence<int, Is...>)
{
    return std::integer_sequence<int, Is..., Val>{};
}

template <int Val, int... Is>
constexpr auto push_front(std::integer_sequence<int, Is...>)
{
    return std::integer_sequence<int, Val, Is...>{};
}

template <int Val, int FirstIndex, int... RestIndices, int... Vals>
constexpr auto fill_impl(std::integer_sequence<int, FirstIndex, RestIndices...> indices,
                         std::integer_sequence<int, Vals...> vals)
{
    if constexpr(sizeof...(RestIndices) == 0)
    {
        return set<FirstIndex, Val>(vals);
    }
    else
    {
        return set<FirstIndex, Val>(
            fill_impl<Val>(std::integer_sequence<int, RestIndices...>{}, vals));
    }
}

template <int Val, int... DstIndices, int... Vals>
constexpr auto fill(std::integer_sequence<int, DstIndices...> dst_indices,
                    std::integer_sequence<int, Vals...>       vals)
{
    if constexpr(sizeof...(DstIndices) == 0)
    {
        return std::integer_sequence<int, Vals...>{};
    }
    else
    {
        return fill_impl<Val>(std::integer_sequence<int, DstIndices...>{},
                              std::integer_sequence<int, Vals...>{});
    }
}


template <int Val, int FirstVal, int... RestVals, int... CurrentVals>
constexpr auto remove_impl(std::integer_sequence<int, CurrentVals...> cur,
                           std::integer_sequence<int, FirstVal, RestVals...>)
{

    if constexpr(sizeof...(RestVals) == 0)
    {
        if constexpr(FirstVal != Val)
        {
            return push_back<FirstVal>(cur);
        }
        else
        {
            return cur;
        }
    }
    else
    {
        if constexpr(FirstVal != Val)
        {
            return remove_impl<Val>(push_back<FirstVal>(cur),
                                    std::integer_sequence<int, RestVals...>{});
        }
        else
        {
            return remove_impl<Val>(cur, std::integer_sequence<int, RestVals...>{});
        }
    }
}

template <int Val, int... Vals>
constexpr auto remove(std::integer_sequence<int, Vals...>)
{
    if constexpr(sizeof...(Vals) == 0)
    {
        return std::integer_sequence<int>{};
    }
    else
    {
        return remove_impl<Val>(std::integer_sequence<int>{},
                                std::integer_sequence<int, Vals...>{});
    }
}

template <int... RemoveIndices, int... Vals>
constexpr auto remove_indices(std::integer_sequence<int, RemoveIndices...> RIs,
                              std::integer_sequence<int, Vals...>          vals)
{
    auto filled = fill<-1>(RIs, vals);
    return remove<-1>(filled);
}

template <typename Impl>
class TensorTraits
{
  public:
    using ImplType      = Impl;
    using ValueType     = typename Impl::ValueType;
    using ShapeSequence = typename Impl::ShapeSequence;

    constexpr static int Dim = ImplType::Dim;


    MUDA_GENERIC TensorTraits() noexcept = default;

    MUDA_GENERIC TensorTraits(const ImplType& impl) noexcept
        : m_impl(impl)
    {
    }

    MUDA_GENERIC ValueType get(const IndexArray<int, Dim>& index) const noexcept
    {
        return m_impl.get(index);
    }

    MUDA_GENERIC void set(ValueType value, const IndexArray<int, Dim>& index) noexcept
    {
        m_impl.set(value, index);
    }

    MUDA_GENERIC int shape(int i) const noexcept { return m_impl.shape(i); }

  private:
    ImplType m_impl;
};

template <class TensorTraits, int... Axis>
class ExprAxes
{
  public:
    using Traits             = TensorTraits;
    using ValueType          = typename Traits::ValueType;
    using ShapeSequence      = typename Traits::ShapeSequence;
    constexpr static int Dim = Traits::Dim;

    using AxisSequence = std::integer_sequence<int, Axis...>;
    using FreeShapeSequence = decltype(remove_indices(AxisSequence{}, ShapeSequence{}));
    using FreeAxisSequence =
        decltype(remove_indices(AxisSequence{}, std::make_integer_sequence<int, Dim>{}));

    constexpr static int AxisCount = AxisSequence::size();
    static_assert((AxisCount > 0 && AxisCount <= Dim), "Invalid axis count");

    MUDA_GENERIC ExprAxes(Traits& impl, std::integer_sequence<int, Axis...>) noexcept
        : m_impl(impl)
        , m_axes{Axis...}
    {
        IndexArray<int, Dim> all_axes;
        for(int i = 0; i < Dim; ++i)
        {
            all_axes[i] = i;
        }

        for(int i = 0; i < AxisCount; ++i)
        {
            auto axis      = m_axes[i];
            all_axes[axis] = -1;
        }

        for(int i = 0, j = 0; i < Dim; ++i)
        {
            if(all_axes[i] != -1)
            {
                m_other_axes[j++] = all_axes[i];
            }
        }
    }

    MUDA_GENERIC Traits&       impl() const noexcept { return m_impl; }
    MUDA_GENERIC int           axis(int i) const noexcept { return m_axes[i]; }
    MUDA_GENERIC constexpr int axis_count() const noexcept { return AxisCount; }
    MUDA_GENERIC int           other_axis(int i) const noexcept
    {
        return m_other_axes[i];
    }
    MUDA_GENERIC constexpr int other_axis_count() const noexcept
    {
        return Dim - AxisCount;
    }

  private:
    Traits&                          m_impl;
    IndexArray<int, AxisCount>       m_axes;
    IndexArray<int, Dim - AxisCount> m_other_axes;
};

// CTAD
template <class TensorTraits, int... Axis>
ExprAxes(TensorTraits&, std::integer_sequence<int, Axis...>)
    -> ExprAxes<TensorTraits, Axis...>;


template <typename Impl>
class TensorRet;

template <typename Impl>
class Tensor
{
  public:
    using Traits             = TensorTraits<Impl>;
    using ValueType          = typename Traits::ValueType;
    using ShapeSequence      = typename Traits::ShapeSequence;
    constexpr static int Dim = Traits::Dim;

    MUDA_GENERIC Tensor() noexcept = default;

    using ImplType = typename Traits::ImplType;

    MUDA_GENERIC Tensor(const Impl& I) noexcept
        : m_impl(I)
    {
    }

    MUDA_GENERIC Tensor(const Tensor& other) noexcept
        : m_impl{other.m_impl}
    {
    }

    template <int... Axis>
    MUDA_GENERIC auto axes() noexcept
    {
        return ExprAxes(m_impl, std::integer_sequence<int, Axis...>{});
    }

    MUDA_GENERIC auto full_axes() noexcept
    {
        return ExprAxes(m_impl, std::make_integer_sequence<int, Dim>{});
    }

    MUDA_GENERIC Traits& impl() noexcept { return m_impl; }

    template <typename... IndexType>
    MUDA_GENERIC ValueType get(IndexType... index) const noexcept
    {
        IndexArray<int, Dim> index_array{index...};
        return m_impl.get(index_array);
    }

    template <typename... IndexType>
    MUDA_GENERIC void set(ValueType value, IndexType... index) noexcept
    {
        IndexArray<int, Dim> index_array{index...};
        m_impl.set(value, index_array);
    }

    MUDA_GENERIC int shape(int i) const noexcept { return m_impl.shape(i); }

    template <typename... IndexType>
    MUDA_GENERIC ValueType operator()(IndexType... index) const noexcept
    {
        return get(index...);
    }

  private:
    Traits m_impl;
};

template <typename AxesL, typename AxesR>
class ExprEinSum
{
  public:
    using TraitsL            = typename AxesL::Traits;
    using ShapeSequenceL     = typename TraitsL::ShapeSequence;
    using FreeShapeSequenceL = typename AxesL::FreeShapeSequence;

    using TraitsR            = typename AxesR::Traits;
    using ShapeSequenceR     = typename TraitsR::ShapeSequence;
    using FreeShapeSequenceR = typename AxesR::FreeShapeSequence;

    using ShapeSequenceD = decltype(concat(FreeShapeSequenceL{}, FreeShapeSequenceR{}));
    using FullShapeSequence = decltype(concat(ShapeSequenceL{}, ShapeSequenceR{}));
    using DstTensor =
        Tensor<ndarray_maker_t<typename TraitsL::ValueType, ShapeSequenceD>>;

    static_assert(AxesL::AxisCount == AxesR::AxisCount, "Axis count mismatch");

    constexpr static int AxisCount = AxesL::AxisCount;
    constexpr static int SrcDimL   = TraitsL::Dim;
    constexpr static int SrcDimR   = TraitsR::Dim;
    constexpr static int DstDim    = SrcDimL + SrcDimR - 2 * AxisCount;

    using ValueType =
        std::common_type_t<typename TraitsL::ValueType, typename TraitsR::ValueType>;

    MUDA_GENERIC ExprEinSum(const AxesL& implL, const AxesR& implR) noexcept
        : m_axesL(implL)
        , m_axesR(implR)
    {
    }

    MUDA_GENERIC void get(ValueType& value, const IndexArray<int, DstDim>& free_index) const
    {
        constexpr int SrcDimL  = TraitsL::Dim;
        constexpr int SrcDimR  = TraitsR::Dim;
        constexpr int DummyDim = AxisCount;

        const TraitsL& L = m_axesL.impl();
        const TraitsR& R = m_axesR.impl();

        IndexArray<int, SrcDimL> indexL;
        IndexArray<int, SrcDimR> indexR;

        build_free_index(free_index, indexL, indexR);

        value = 0.0;

        nested_dummy_sum<0, DummyDim>(value, indexL, indexR);
    }

    template <class TensorTraits, int... Axis>
    MUDA_GENERIC void eval_to(const ExprAxes<TensorTraits, Axis...>& dst) const
    {
        using DstAxes   = ExprAxes<TensorTraits, Axis...>;
        using TraitsDst = typename DstAxes::Traits;
        TraitsDst& D    = dst.impl();

        IndexArray<int, DstDim> indexD;
        IndexArray<int, DstDim> free_index;
        nested_free_eval<DstAxes, 0, DstDim>(dst, free_index, indexD);
    }

    template <class Impl>
    MUDA_GENERIC void eval_to(Tensor<Impl>& dst) const
    {
        auto axesD = dst.full_axes();
        eval_to(axesD);
    }

    MUDA_GENERIC auto eval() const
    {
        DstTensor result;
        eval_to(result);
        return result;
    }

    MUDA_GENERIC operator DstTensor() const { return eval(); }

  private:
    template <typename DstAxes, int I, int COUNT>
    inline MUDA_GENERIC void nested_free_eval(const DstAxes& axesD,
                                              IndexArray<int, DstDim>& free_index,
                                              IndexArray<int, DstDim>& indexD) const
    {
        auto& D = axesD.impl();

        if constexpr(I < COUNT)
        {
            int AxisD_I  = axesD.axis(I);
            int CountD_I = D.shape(AxisD_I);

            for(int i = 0; i < CountD_I; ++i)
            {
                indexD[AxisD_I] = i;
                free_index[I]   = i;

                nested_free_eval<DstAxes, I + 1, COUNT>(axesD, free_index, indexD);
            }
        }
        else
        {
            ValueType value;
            get(value, free_index);
            D.set(value, indexD);
        }
    }

    template <int I, int COUNT>
    inline MUDA_GENERIC void nested_dummy_sum(ValueType&               value,
                                              IndexArray<int, SrcDimL> indexL,
                                              IndexArray<int, SrcDimR> indexR) const
    {
        auto& L = m_axesL.impl();
        auto& R = m_axesR.impl();

        if constexpr(I < COUNT)
        {
            int AxisL_I = m_axesL.axis(I);
            int AxisR_I = m_axesR.axis(I);

            int CountL_I = L.shape(AxisL_I);
            int CountR_I = R.shape(AxisR_I);

            MUDA_ASSERT(CountL_I == CountR_I, "Axis size mismatch");

            // dummy index sum
            for(int i = 0; i < CountL_I; ++i)
            {
                indexL[AxisL_I] = i;
                indexR[AxisR_I] = i;

                nested_dummy_sum<I + 1, COUNT>(value, indexL, indexR);
            }
        }
        else
        {
            value += L.get(indexL) * R.get(indexR);
        }
    }


    MUDA_GENERIC
    void build_free_index(const IndexArray<int, DstDim>& free_index,
                          IndexArray<int, SrcDimL>&      indexL,
                          IndexArray<int, SrcDimR>&      indexR) const
    {
        constexpr static int FreeAxisCountL = SrcDimL - AxisCount;
        constexpr static int FreeAxisCountR = SrcDimR - AxisCount;

        static_assert(DstDim == FreeAxisCountL + FreeAxisCountR, "Dimension mismatch");

        if(DstDim == 0)
            return;

        for(int i = 0; i < DstDim; ++i)
        {
            if(i < FreeAxisCountL)
            {
                indexL[m_axesL.other_axis(i)] = free_index[i];
            }
            else
            {
                indexR[m_axesR.other_axis(i - FreeAxisCountL)] = free_index[i];
            }
        }
    }

    const AxesL& m_axesL;
    const AxesR& m_axesR;
};

template <class TensorTraitsL, class TensorTraitsR, int... AxisL, int... AxisR>
MUDA_GENERIC auto operator*(ExprAxes<TensorTraitsL, AxisL...>&& lhs,
                            ExprAxes<TensorTraitsR, AxisR...>&& rhs) noexcept
{
    static_assert(sizeof...(AxisL) == sizeof...(AxisR), "Axis count mismatch");
    return ExprEinSum{lhs, rhs};
}

template <typename AxesL, typename AxesR>
class TensorRet<ExprEinSum<AxesL, AxesR>>
{
  public:
    using ExprType = ExprEinSum<AxesL, AxesR>;

    MUDA_GENERIC TensorRet(const ExprType& expr) noexcept
        : m_expr(expr)
    {
    }

    template <typename Impl>
    MUDA_GENERIC void eval_to(Tensor<Impl>& dst) const
    {
        m_expr.eval_to(dst);
    }

  private:
    const ExprType& m_expr;
};
}  // namespace muda::einops

#endif // _MSC_VER