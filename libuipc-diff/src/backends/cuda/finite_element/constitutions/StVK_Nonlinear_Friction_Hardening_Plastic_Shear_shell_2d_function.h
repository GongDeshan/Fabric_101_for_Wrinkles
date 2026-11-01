#pragma once
#include <muda/ext/eigen/eigen_core_cxx20.h>

namespace uipc::backend::cuda
{
namespace sym::stvk_nonlinear_friction_hardening_plastic_shear_2d
{
	#include "sym/stvk_nonlinear_friction_plastic_shear_shell_2d.inl"
    
    inline UIPC_GENERIC Float sign(Float& value)
    {
        if(value > 0.0)
            return 1.0;
        else if(value < 0.0)
            return -1.0;
        else
            return 0.0;
    }
}
}  // namespace uipc::backend::cuda
