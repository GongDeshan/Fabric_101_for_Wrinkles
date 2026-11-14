#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <utils/matrix_assembly_utils.h>
#include <cmath>

namespace uipc::backend::cuda
{
class TriangleWind final : public FiniteElementExtraConstitution
{
  public:
    using FiniteElementExtraConstitution::FiniteElementExtraConstitution;
  private:
    FiniteElementMethod* fem=nullptr;
    vector<geometry::SimplicialComplex*> geometries;
    vector<Vector3i> host_indices;
    vector<Vector3> host_velocity;
    vector<Float> host_density,host_normal,host_tangent;
    vector<IndexT> host_enabled;
    muda::DeviceBuffer<Vector3i> indices;
    muda::DeviceBuffer<Vector3> velocity,force;
    muda::DeviceBuffer<Float> density,normal_drag,tangent_drag;
    muda::DeviceBuffer<IndexT> enabled;

  public:
    U64 get_uid() const noexcept override { return 1069; }
    void do_build(BuildInfo&) override
    {
        fem=&require<FiniteElementMethod>();
        require<DofPredictor>().on_predict(*this,[this](DofPredictor::PredictInfo&){ update(); });
    }
    void upload()
    {
        host_indices.clear();host_velocity.clear();host_density.clear();
        host_normal.clear();host_tangent.clear();host_enabled.clear();
        for(auto* sc:geometries)
        {
            auto offset=sc->meta().find<IndexT>(builtin::backend_fem_vertex_offset)->view().front();
            auto triangles=sc->triangles().topo().view();
            auto velocities=sc->triangles().find<Vector3>("wind/velocity")->view();
            auto densities=sc->triangles().find<Float>("wind/air_density")->view();
            auto normals=sc->triangles().find<Float>("wind/normal_drag")->view();
            auto tangents=sc->triangles().find<Float>("wind/tangential_drag")->view();
            auto enables=sc->triangles().find<IndexT>("wind/enabled")->view();
            for(SizeT i=0;i<triangles.size();++i)
            {
                UIPC_ASSERT(velocities[i].allFinite() && std::isfinite(densities[i])
                    && std::isfinite(normals[i]) && std::isfinite(tangents[i])
                    && densities[i]>=0 && normals[i]>=0 && tangents[i]>=0,
                    "Invalid TriangleWind attribute");
                Vector3i t=triangles[i];t.array()+=offset;
                host_indices.push_back(t);host_velocity.push_back(velocities[i]);
                host_density.push_back(densities[i]);host_normal.push_back(normals[i]);
                host_tangent.push_back(tangents[i]);host_enabled.push_back(enables[i]);
            }
        }
        const auto count=host_indices.size();
        UIPC_ASSERT(count==force.size(),"TriangleWind topology must remain fixed after world.init");
        if(count==0) return;
        indices.view().copy_from(host_indices.data());velocity.view().copy_from(host_velocity.data());
        density.view().copy_from(host_density.data());normal_drag.view().copy_from(host_normal.data());
        tangent_drag.view().copy_from(host_tangent.data());enabled.view().copy_from(host_enabled.data());
    }
    void do_init(FilteredInfo& info) override
    {
        UIPC_ASSERT(!world().scene().info()["diff_sim"]["enable"].get<bool>(),
                    "TriangleWind currently supports forward simulation only");
        geometries.clear();
        SizeT count=0;
        info.for_each(world().scene().geometries(),
            [&](const FiniteElementMethod::ForEachInfo&,geometry::SimplicialComplex& sc)
            { geometries.push_back(&sc);count+=sc.triangles().size(); });
        indices.resize(count);velocity.resize(count);force.resize(count);
        density.resize(count);normal_drag.resize(count);tangent_drag.resize(count);enabled.resize(count);
        upload();
        force.fill(Vector3::Zero().eval());
    }
    void update()
    {
        upload();
        if(force.size()==0) return;
        using namespace muda;
        ParallelFor().file_line(__FILE__,__LINE__).apply(force.size(),
            [ts=indices.cviewer(),wind=velocity.cviewer(),rho=density.cviewer(),
             cn=normal_drag.cviewer(),ct=tangent_drag.cviewer(),on=enabled.cviewer(),
             xs=fem->xs().cviewer(),vs=fem->vs().cviewer(),fs=force.viewer()] __device__(int i)
            {
                fs(i)=Vector3::Zero();
                if(!on(i)) return;
                const auto t=ts(i);
                Vector3 cross=(xs(t(1))-xs(t(0))).cross(xs(t(2))-xs(t(0)));
                const Float norm=cross.norm();
                if(norm<=1.0e-20) return;
                Vector3 n=cross/norm;
                Vector3 rel=wind(i)-(vs(t(0))+vs(t(1))+vs(t(2)))/3.0;
                Float vn=rel.dot(n);
                Vector3 vt=rel-vn*n;
                fs(i)=0.25*rho(i)*norm*(cn(i)*vn*std::abs(vn)*n+ct(i)*vt.norm()*vt);
            });
    }
    void do_report_extent(ReportExtentInfo& info) override
    { info.energy_count(force.size());info.stencil_dim(3); }
    void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        ParallelFor().file_line(__FILE__,__LINE__).apply(force.size(),
            [ts=indices.cviewer(),fs=force.cviewer(),xs=info.xs().cviewer(),
             fixed=fem->is_fixed().cviewer(),es=info.energies().viewer(),dt=info.dt()] __device__(int i)
            {
                const auto t=ts(i);Float energy=0;
                for(int j=0;j<3;++j) if(!fixed(t(j))) energy-=fs(i).dot(xs(t(j)))/3.0;
                es(i)=dt*dt*energy;
            });
    }
    void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        UIPC_ASSERT(!info.is_adjoint(),"TriangleWind does not implement trajectory adjoints");
        using namespace muda;
        ParallelFor().file_line(__FILE__,__LINE__).apply(force.size(),
            [ts=indices.cviewer(),fs=force.cviewer(),fixed=fem->is_fixed().cviewer(),
             gs=info.gradients().viewer(),hs=info.hessians().viewer(),dt=info.dt()] __device__(int i) mutable
            {
                const auto t=ts(i);Vector9 g=Vector9::Zero();
                for(int j=0;j<3;++j) if(!fixed(t(j))) g.segment<3>(3*j)=-dt*dt*fs(i)/3.0;
                Matrix9x9 h=Matrix9x9::Zero();
                assemble<3>(gs,3*i,t,g);assemble<3>(hs,9*i,t,h);
            });
    }
};
REGISTER_SIM_SYSTEM(TriangleWind);
}
