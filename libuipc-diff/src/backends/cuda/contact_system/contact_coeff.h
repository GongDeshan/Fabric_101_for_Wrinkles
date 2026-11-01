#pragma once
#include <type_define.h>

namespace uipc::backend::cuda
{
class ContactCoeff
{
  public:
    // normal stiffness
    Float kappa = 0.0;
    // friction coefficient
    Float mu = 0.0;
};

// Used in DiffSim:
// Related to Attribute `diff/resistance` `diff/friction_rate`
class DiffContactCoeff
{
  public:
    // normal stiffness parm id
    IndexT diff_kappa = -1;
    // friction coefficient parm id
    IndexT diff_mu = -1;
};
}  // namespace uipc::backend::cuda
