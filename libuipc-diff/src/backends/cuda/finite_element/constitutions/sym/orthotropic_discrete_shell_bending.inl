
template <typename T>
__host__ __device__ void E(T&       R,
                           const T& kappa_warp,
                           const T& kappa_weft,
                           const T& kappa_torsion,
                           const T& kappa_poisson,
                           const T& theta_warp,
                           const T& theta_weft,
                           const T& theta_torsion,
                           const T& L0,
                           const T& h_bar)
{
    R = L0
        * (kappa_warp * std::pow(theta_warp, 2) + kappa_weft * std::pow(theta_weft, 2)
           + 2.0 * kappa_poisson * theta_warp * theta_weft
           + kappa_torsion * std::pow(theta_torsion, 2))
        / h_bar;
}

template <typename T>
__host__ __device__ void E_friction(T&       R,
                                    const T& k_warp,
                                    const T& k_weft,
                                    const T& theta_var_warp,
                                    const T& theta_var_weft,
                                    const T& L0,
                                    const T& h_bar)
{
    R = L0 * (k_warp * std::pow(theta_var_warp, 2) + k_weft * std::pow(theta_var_weft, 2)) / h_bar;
}


template <typename T>
__host__ __device__ void dEdtheta(T&       R,
                                  const T& kappa_warp,
                                  const T& kappa_weft,
                                  const T& kappa_torsion,
                                  const T& kappa_poisson,
                                  const T& theta_warp,
                                  const T& theta_weft,
                                  const T& theta_torsion,
                                  const T& cos_angle,
                                  const T& sin_angle,
                                  const T& L0,
                                  const T& h_bar)
{
    R = 2.0 * L0
        * (kappa_warp * theta_warp * std::pow(cos_angle, 2)
           + kappa_weft * theta_weft * std::pow(sin_angle, 2)
           + kappa_poisson
                 * (theta_warp * std::pow(sin_angle, 2) + theta_weft * std::pow(cos_angle, 2))
           + kappa_torsion * theta_torsion * sin_angle * cos_angle)
        / h_bar;
}

template <typename T>
__host__ __device__ void dE_friction_dtheta(T&       R,
                                            const T& k_warp,
                                            const T& k_weft,
                                            const T& cos_angle,
                                            const T& sin_angle,
                                            const T& theta_var_warp,
                                            const T& theta_var_weft,
                                            const T& L0,
                                            const T& h_bar)
{
    R = 2.0 * L0
        * (k_warp * theta_var_warp * std::pow(cos_angle, 2)
           + k_weft * theta_var_weft * std::pow(sin_angle, 2))
        / h_bar;
}


template <typename T>
__host__ __device__ void ddEddtheta(T&       R,
                                    const T& kappa_warp,
                                    const T& kappa_weft,
                                    const T& kappa_torsion,
                                    const T& kappa_poisson,
                                    const T& cos_angle,
                                    const T& sin_angle,
                                    const T& L0,
                                    const T& h_bar)
{
    R = 2.0 * L0
        * (kappa_warp * std::pow(cos_angle, 4) + kappa_weft * std::pow(sin_angle, 4)
           + 2 * kappa_poisson * std::pow(sin_angle, 2) * std::pow(cos_angle, 2)
           + kappa_torsion * std::pow(sin_angle, 2) * std::pow(cos_angle, 2))
        / h_bar;
}

template <typename T>
__host__ __device__ void ddE_friction_ddtheta(T&       R,
                                              const T& k_warp,
                                              const T& k_weft,
                                              const T& cos_angle,
                                              const T& sin_angle,
                                              const T& L0,
                                              const T& h_bar)
{
    R = 2.0 * L0 * (k_warp * std::pow(cos_angle, 4) + k_weft * std::pow(sin_angle, 4)) / h_bar;
}
