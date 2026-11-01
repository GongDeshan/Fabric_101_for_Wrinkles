#pragma once
#include <Eigen/Core>
#include <cmath>

#if defined(__CUDACC__)
#define UIPC_YARN_HD __host__ __device__
#else
#define UIPC_YARN_HD
#endif

namespace uipc::backend::cuda::yarn_rotation
{


template<class T> struct Jet
{
    T value;
    Eigen::Matrix<T,4,1> gradient;
    Eigen::Matrix<T,4,4> hessian;
    UIPC_YARN_HD Jet(T v = T(0))
        : value(v), gradient(Eigen::Matrix<T,4,1>::Zero()),
          hessian(Eigen::Matrix<T,4,4>::Zero()) {}
    UIPC_YARN_HD static Jet variable(T v, int coordinate)
    {
        Jet r(v);
        if(coordinate >= 0) r.gradient(coordinate) = T(1);
        return r;
    }
};
template<class T> UIPC_YARN_HD Jet<T> operator+(const Jet<T>& a, const Jet<T>& b)
{
    Jet<T> r(a.value+b.value);
    r.gradient=a.gradient+b.gradient; r.hessian=a.hessian+b.hessian;
    return r;
}
template<class T> UIPC_YARN_HD Jet<T> operator-(const Jet<T>& a, const Jet<T>& b)
{
    Jet<T> r(a.value-b.value);
    r.gradient=a.gradient-b.gradient; r.hessian=a.hessian-b.hessian;
    return r;
}
template<class T> UIPC_YARN_HD Jet<T> operator*(const Jet<T>& a, const Jet<T>& b)
{
    Jet<T> r(a.value*b.value);
    r.gradient=a.gradient*b.value+b.gradient*a.value;
    r.hessian=a.hessian*b.value+b.hessian*a.value
              +a.gradient*b.gradient.transpose()+b.gradient*a.gradient.transpose();
    return r;
}
template<class T> UIPC_YARN_HD Jet<T> reciprocal(const Jet<T>& a)
{
    Jet<T> r(T(1)/a.value);
    const T d=-r.value*r.value, dd=T(2)*r.value*r.value*r.value;
    r.gradient=d*a.gradient;
    r.hessian=d*a.hessian+dd*(a.gradient*a.gradient.transpose());
    return r;
}
template<class T> UIPC_YARN_HD Jet<T> operator/(const Jet<T>& a, const Jet<T>& b)
{
    return a*reciprocal(b);
}
template<class T> UIPC_YARN_HD Jet<T> absolute(const Jet<T>& a)
{
    const T sign=a.value>T(0)?T(1):(a.value<T(0)?T(-1):T(0));
    Jet<T> r(std::abs(a.value));
    r.gradient=sign*a.gradient; r.hessian=sign*a.hessian;
    return r;
}

template<class T> UIPC_YARN_HD Jet<T> cosine_squared(
    const Jet<T>& guu, const Jet<T>& gvv, const Jet<T>& guv)
{



    const Jet<T> one(T(1)), two(T(2)), four(T(4));
    const Jet<T> denominator=(one+two*guu)*(one+two*gvv);
    if(denominator.value<=T(1e-24)) return Jet<T>(T(0));
    Jet<T> t=four*guv*guv/denominator;


    if(t.value<T(0)) return Jet<T>(T(0));
    if(t.value>T(1)-T(1e-12)) return Jet<T>(T(1)-T(1e-12));
    return t;
}



template<class T> UIPC_YARN_HD Jet<T> energy_in_strain(
    const Eigen::Matrix<T,3,1>& G, const Eigen::Matrix<T,9,1>& parameters,
    T plastic_uu,T plastic_vv,T plastic_uv,T anchor_uv,bool rotate,
    int differentiated_parameter=-1,T dplastic=T(0),T danchor=T(0))
{
    using J=Jet<T>;
    const J guu=J::variable(G(0),0), gvv=J::variable(G(1),1), q=J::variable(G(2),2);
    J p[9];
    for(int i=0;i<9;++i) p[i]=J::variable(parameters(i),i==differentiated_parameter?3:-1);
    const J k=p[3]+p[4]*absolute(q);
    J k11=p[0], k22=p[1], k12=p[2], k33=k;
    if(rotate)
    {
        const J t=cosine_squared(guu,gvv,q), u=J(T(1))-t;


        k11=p[0]+J(T(2))*p[2]*t+p[1]*t*t+J(T(4))*k*t;
        k22=p[1]*u*u;
        k12=p[2]*u+p[1]*t*u;
        k33=p[1]*t*u+k*u;
    }
    J plastic(plastic_uv),anchor(anchor_uv);
    plastic.gradient(3)=dplastic;anchor.gradient(3)=danchor;
    const J eu=guu-J(plastic_uu), ev=gvv-J(plastic_vv);
    const J eq=q-plastic, friction=q-anchor;
    return J(T(.5))*k11*eu*eu+k12*eu*ev+J(T(.5))*k22*ev*ev
         + J(T(.5))*k33*eq*eq+J(T(.5))*p[5]*friction*friction;
}

template<class T> struct Geometry
{
    Eigen::Matrix<T,3,1> strain;
    Eigen::Matrix<T,3,9> jacobian;
    Eigen::Matrix<T,9,9> huu,hvv,huv;
};
template<class T> UIPC_YARN_HD Geometry<T> geometry(
    const Eigen::Matrix<T,9,1>& x,const Eigen::Matrix<T,6,1>& X)
{
    Eigen::Matrix<T,2,2> dm,inv;
    dm.col(0)=X.template segment<2>(2)-X.template segment<2>(0);
    dm.col(1)=X.template segment<2>(4)-X.template segment<2>(0);
    const T det=dm(0,0)*dm(1,1)-dm(0,1)*dm(1,0);
    inv(0,0)=dm(1,1)/det;inv(0,1)=-dm(0,1)/det;
    inv(1,0)=-dm(1,0)/det;inv(1,1)=dm(0,0)/det;
    Eigen::Matrix<T,3,2> ds;
    ds.col(0)=x.template segment<3>(3)-x.template segment<3>(0);
    ds.col(1)=x.template segment<3>(6)-x.template segment<3>(0);
    const Eigen::Matrix<T,3,2> F=ds*inv;
    Eigen::Matrix<T,9,3> du,dv;
    const Eigen::Matrix<T,3,3> id=Eigen::Matrix<T,3,3>::Identity();
    du.template block<3,3>(0,0)=(-inv(0,0)-inv(1,0))*id;
    du.template block<3,3>(3,0)=inv(0,0)*id;
    du.template block<3,3>(6,0)=inv(1,0)*id;
    dv.template block<3,3>(0,0)=(-inv(0,1)-inv(1,1))*id;
    dv.template block<3,3>(3,0)=inv(0,1)*id;
    dv.template block<3,3>(6,0)=inv(1,1)*id;
    Geometry<T> r;
    r.strain << T(.5)*(F.col(0).squaredNorm()-T(1)),
                T(.5)*(F.col(1).squaredNorm()-T(1)),T(.5)*F.col(0).dot(F.col(1));
    r.jacobian.row(0)=(du*F.col(0)).transpose();
    r.jacobian.row(1)=(dv*F.col(1)).transpose();
    r.jacobian.row(2)=(T(.5)*(du*F.col(1)+dv*F.col(0))).transpose();
    r.huu=du*du.transpose();r.hvv=dv*dv.transpose();
    r.huv=T(.5)*(du*dv.transpose()+dv*du.transpose());
    return r;
}

template<class T> struct Evaluation
{
    T energy;
    Eigen::Matrix<T,9,1> gradient,parameter_gradient;
    Eigen::Matrix<T,9,9> hessian;
};
template<class T> UIPC_YARN_HD Evaluation<T> evaluate(
    const Eigen::Matrix<T,9,1>& x,const Eigen::Matrix<T,6,1>& X,
    const Eigen::Matrix<T,9,1>& parameters,T pu,T pv,T pq,T aq,bool rotate,
    int differentiated_parameter=-1,T dplastic=T(0),T danchor=T(0))
{
    const auto g=geometry(x,X);
    const auto e=energy_in_strain(g.strain,parameters,pu,pv,pq,aq,rotate,
                                  differentiated_parameter,dplastic,danchor);
    Evaluation<T> r;
    r.energy=e.value;
    r.gradient=g.jacobian.transpose()*e.gradient.template head<3>();
    r.hessian=g.jacobian.transpose()*e.hessian.template topLeftCorner<3,3>()*g.jacobian
             +e.gradient(0)*g.huu+e.gradient(1)*g.hvv+e.gradient(2)*g.huv;
    r.parameter_gradient=g.jacobian.transpose()*e.hessian.template block<3,1>(0,3);
    return r;
}

template<class T> UIPC_YARN_HD T effective_shear(
    T gu,T gv,T q,T k22,T k0,T k1,bool rotate)
{
    if(!rotate) return k0+k1*std::abs(q);
    const T t=cosine_squared(Jet<T>(gu),Jet<T>(gv),Jet<T>(q)).value,u=T(1)-t;
    return k22*t*u+(k0+k1*std::abs(q))*u;
}
} // namespace uipc::backend::cuda::yarn_rotation
#undef UIPC_YARN_HD
