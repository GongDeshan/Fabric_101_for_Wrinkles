#ifdef _MSC_VER
#include <iostream>
#include <utils/einops/tensor.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <muda/launch.h>
#include <muda/logger.h>
#include <muda/ext/eigen.h>
#include <muda/ext/eigen/log_proxy.h>


using namespace muda::einops;

// print integer_sequence
template <int... Is>
void print(std::integer_sequence<int, Is...>)
{
    ((std::cout << Is << " "), ...);
    std::cout << std::endl;
}

template <typename Impl>
void print_tensor(const Tensor<Impl>& t)
{
    using T            = Tensor<Impl>;
    constexpr auto Dim = T::Dim;
    if constexpr(Dim == 0)
    {
        std::cout << t() << std::endl;
    }

    if constexpr(Dim == 1)
    {
        for(int i = 0; i < t.shape(0); ++i)
        {
            std::cout << t(i) << " ";
        }
        std::cout << std::endl;
    }

    if constexpr(Dim == 2)
    {
        for(int i = 0; i < t.shape(0); ++i)
        {
            for(int j = 0; j < t.shape(1); ++j)
            {
                std::cout << t(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    if constexpr(Dim == 3)
    {
        for(int i = 0; i < t.shape(0); ++i)
        {
            for(int j = 0; j < t.shape(1); ++j)
            {
                for(int k = 0; k < t.shape(2); ++k)
                {
                    std::cout << t(i, j, k) << std::endl;
                }
                std::cout << "----------------" << std::endl;
            }
            std::cout << "================" << std::endl;
        }
    }

    std::cout << std::endl;
}

int main()
{
    using Tensor2x2x2     = Tensor<NDArray<double, 2, 2, 2>>;
    using Tensor2x2       = Tensor<NDArray<double, 2, 2>>;
    using Tensor2         = Tensor<NDArray<double, 2>>;
    using Tensor0         = Tensor<Scalar<double>>;
    using TensorView2x2   = Tensor<NDArrayView<double, 2, 2>>;
    using TensorView2x2x2 = Tensor<NDArrayView<double, 2, 2, 2>>;

    //std::vector<double> data(4);
    //std::vector<double> data_B(4);
    //std::iota(data.begin(), data.end(), 0);
    //NDArrayView<double, 2, 2> array2x2{data.data()};
    //NDArrayView<double, 2, 2> array2x2_B{data_B.data()};

    //auto A = TensorView2x2{array2x2};
    //print_tensor(A);
    //auto B = TensorView2x2{array2x2_B};

    //{
    //    auto&   A_ij = A;
    //    Tensor0 s1   = A_ij.axes<0, 1>() * A_ij.axes<1, 0>();
    //    print_tensor(s1);
    //    Tensor0 s2 = A_ij.axes<0, 1>() * A_ij.axes<0, 1>();
    //    print_tensor(s2);
    //}

    //{
    //    auto& A_ij = A;
    //    auto& A_jk = A;
    //    auto& A_ik = A;
    //    auto& A_lj = A;

    //    Tensor2x2 B_ik = A_ij.axes<1>() * A_jk.axes<0>();  // == A @ A
    //    print_tensor(B_ik);

    //    Tensor2x2 B_jk = A_ij.axes<0>() * A_ik.axes<0>();  // == A.T @ A
    //    print_tensor(B_jk);

    //    Tensor2x2 B_il = A_ij.axes<1>() * A_lj.axes<1>();  // == A @ A.T
    //    print_tensor(B_il);
    //}

    {
        std::vector<double> data(12);

        data[0] = 111;
        data[1] = 211;
        data[2] = 121;
        data[3] = 221;
        data[4] = 131;
        data[5] = 231;

        data[6]  = 112;
        data[7]  = 212;
        data[8]  = 122;
        data[9]  = 222;
        data[10] = 132;
        data[11] = 232;

        NDArrayView<double, 2, 3, 2> ndarray{data.data()};
        using Tensor2x3x2 = Tensor<NDArrayView<double, 2, 3, 2>>;
        auto A            = Tensor2x3x2{ndarray};

        print_tensor(A);
    }

    {
        Eigen::Matrix2d m;
        m << 1, 2, 3, 4;
        TensorView2x2 m_view = array_view(m);
        print_tensor(m_view);

        Eigen::Matrix2d n;
        TensorView2x2   n_view = array_view(n);

        auto expr = m_view.axes<1>() * m_view.axes<0>();

        expr.eval_to(n_view);
        print_tensor(n_view);

        std::cout << m * m << std::endl << std::endl;
    }

    Tensor2x2x2 V;
    V.set(1, 0, 0, 0);
    V.set(2, 0, 0, 1);
    V.set(3, 0, 1, 0);
    V.set(4, 0, 1, 1);
    V.set(5, 1, 0, 0);
    V.set(6, 1, 0, 1);
    V.set(7, 1, 1, 0);
    V.set(8, 1, 1, 1);
    print_tensor(V);

    Tensor2 T;
    T.set(1, 0);
    T.set(2, 1);
    print_tensor(T);

    {
        Eigen::Matrix2d m;
        TensorView2x2   m_view = array_view(m);

        // V_ijk * T_j
        auto expr = V.axes<1>() * T.axes<0>();
        expr.eval_to(m_view);

        Eigen::Matrix2d real_m;
        real_m(0, 0) = V(0, 0, 0) * T(0) + V(0, 1, 0) * T(1);
        real_m(0, 1) = V(0, 0, 1) * T(0) + V(0, 1, 1) * T(1);
        real_m(1, 0) = V(1, 0, 0) * T(0) + V(1, 1, 0) * T(1);
        real_m(1, 1) = V(1, 0, 1) * T(0) + V(1, 1, 1) * T(1);

        std::cout << "m: \n" << m << std::endl;
        std::cout << "real_m: \n" << real_m << std::endl;
    }

    std::cout << std::endl;

    {
        Eigen::Matrix2d m;
        TensorView2x2   m_view = array_view(m);

        // V_ijk * T_i
        auto expr = V.axes<0>() * T.axes<0>();
        expr.eval_to(m_view);

        Eigen::Matrix2d real_m;
        real_m(0, 0) = V(0, 0, 0) * T(0) + V(1, 0, 0) * T(1);
        real_m(0, 1) = V(0, 0, 1) * T(0) + V(1, 0, 1) * T(1);
        real_m(1, 0) = V(0, 1, 0) * T(0) + V(1, 1, 0) * T(1);
        real_m(1, 1) = V(0, 1, 1) * T(0) + V(1, 1, 1) * T(1);

        std::cout << "m: \n" << m << std::endl;
        std::cout << "real_m: \n" << real_m << std::endl;
    }

    std::cout << std::endl;

    using namespace muda;

    Logger logger;

    Launch()
        .apply(
            [logger = logger.viewer()] __device__() mutable
            {
                Tensor2x2x2 V;
                V.set(1, 0, 0, 0);
                V.set(2, 0, 0, 1);
                V.set(3, 0, 1, 0);
                V.set(4, 0, 1, 1);
                V.set(5, 1, 0, 0);
                V.set(6, 1, 0, 1);
                V.set(7, 1, 1, 0);
                V.set(8, 1, 1, 1);

                Tensor2 T;
                T.set(1, 0);
                T.set(2, 1);

                Eigen::Matrix2d m;
                TensorView2x2   m_view = array_view(m);

                auto expr = V.axes<1>() * T.axes<0>();
                expr.eval_to(m_view);

                logger << "kernel m: \n" << m << "\n";
            })
        .wait();

    logger.retrieve();
}
#else

int main() {}

#endif  // _MSC_VER
