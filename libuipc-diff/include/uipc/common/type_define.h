#pragma once
#include <Eigen/Core>
namespace uipc
{
using Float  = double;
using IndexT = int32_t;
using SizeT  = std::size_t;
using I32    = int32_t;
using I64    = int64_t;
using U32    = uint32_t;
using U64    = uint64_t;

using Eigen::Matrix;
using Eigen::Vector;

using Vector2 = Vector<Float, 2>;
using Vector3 = Vector<Float, 3>;
using Vector4 = Vector<Float, 4>;

using Vector2i = Vector<int32_t, 2>;
using Vector3i = Vector<int32_t, 3>;
using Vector4i = Vector<int32_t, 4>;

using Vector6  = Vector<Float, 6>;
using Vector9  = Vector<Float, 9>;
using Vector12 = Vector<Float, 12>;

using Vector6i  = Vector<int32_t, 6>;
using Vector9i  = Vector<int32_t, 9>;
using Vector12i = Vector<int32_t, 12>;

using Matrix2x2 = Matrix<Float, 2, 2>;
using Matrix3x3 = Matrix<Float, 3, 3>;
using Matrix4x4 = Matrix<Float, 4, 4>;

using Matrix2x2i = Matrix<int32_t, 2, 2>;
using Matrix3x3i = Matrix<int32_t, 3, 3>;
using Matrix4x4i = Matrix<int32_t, 4, 4>;

using Matrix6x6   = Matrix<Float, 6, 6>;
using Matrix9x9   = Matrix<Float, 9, 9>;
using Matrix12x12 = Matrix<Float, 12, 12>;

using Matrix6x6i   = Matrix<int32_t, 6, 6>;
using Matrix9x9i   = Matrix<int32_t, 9, 9>;
using Matrix12x12i = Matrix<int32_t, 12, 12>;

using Transform   = Eigen::Transform<Float, 3, Eigen::Affine>;
using Translation = Eigen::Translation<Float, 3>;

using VectorX    = Eigen::VectorX<Float>;
using VectorXi   = Eigen::VectorX<IndexT>;
using VectorXi64 = Eigen::VectorX<I64>;
using VectorXu   = Eigen::VectorX<U32>;
using VectorXu64 = Eigen::VectorX<U64>;

using MatrixX  = Eigen::MatrixX<Float>;
using MatrixXi = Eigen::MatrixX<IndexT>;
}  // namespace uipc
