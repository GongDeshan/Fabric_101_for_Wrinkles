#pragma once
#include "stvk_yarn_rotation_math.h"
#if defined(__CUDACC__)
#define UIPC_YARN_HISTORY_HD __host__ __device__
#else
#define UIPC_YARN_HISTORY_HD
#endif
namespace uipc::backend::cuda::yarn_rotation
{

template<class T> struct HistoryTransition
{
    Eigen::Matrix<T,3,3> state;
    Eigen::Matrix<T,3,3> strain;
};

template<class T> UIPC_YARN_HISTORY_HD HistoryTransition<T> history_transition(
    const Eigen::Matrix<T,3,1>& G,T plastic,T hardening,T anchor,
    T k22,T k0,T k1,T kh,T yield0,T threshold,bool rotate)
{
    using J=Jet<T>;
    const J gu=J::variable(G(0),0),gv=J::variable(G(1),1),qj=J::variable(G(2),2);
    J effective=J(k0)+J(k1)*absolute(qj);
    if(rotate)
    {
        const J t=cosine_squared(gu,gv,qj),u=J(T(1))-t;
        effective=J(k22)*t*u+effective*u;
    }
    const T k=effective.value>T(1e-12)?effective.value:T(1e-12);
    Eigen::Matrix<T,3,1> dk=Eigen::Matrix<T,3,1>::Zero();
    if(effective.value>T(1e-12))dk=effective.gradient.template head<3>();
    const T q=G(2),elastic=q-plastic;
    const T sign=elastic>T(0)?T(1):(elastic<T(0)?T(-1):T(0));
    const T excess=std::abs(elastic)-yield0-hardening*kh/k;
    HistoryTransition<T> r;
    r.state.setIdentity();r.strain.setZero();
    if(excess>T(0))
    {
        const T ratio=k/(k+kh),complement=kh/(k+kh);
        r.state(0,0)=complement;r.state(0,1)=-sign*complement;
        r.state(1,0)=-sign*ratio;r.state(1,1)=ratio;
        const T dk_factor=hardening*kh/(k*k)*ratio+excess*kh/((k+kh)*(k+kh));
        Eigen::Matrix<T,3,1> increment=dk_factor*dk;
        increment(2)+=sign*ratio;
        r.strain.row(0)=(sign*increment).transpose();
        r.strain.row(1)=increment.transpose();
    }
    if(std::abs(q-anchor)>threshold)
    {
        r.state(2,2)=T(0);r.strain(2,2)=T(1);
    }
    return r;
}
}
#undef UIPC_YARN_HISTORY_HD
