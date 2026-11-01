#pragma once
#include <sim_system.h>
#include <contact_system/global_diff_contact_manager.h>

namespace uipc::backend::cuda
{
class ContactDiffDofReporter : public SimSystem
{
    friend class GlobalDiffContactManager;

  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    using DiffDofExtentInfo = GlobalDiffContactManager::ContactDiffDofExtentInfo;
    using DiffDofInfo = GlobalDiffContactManager::ContactDiffDofInfo;

  protected:
    virtual void do_report_extent(DiffDofExtentInfo& info) = 0;
    virtual void do_build(BuildInfo& info)                 = 0;
    virtual void do_assemble(DiffDofInfo& info)            = 0;
    virtual void do_init()                                 = 0;

  private:
    virtual void do_build() final override;
};
}  // namespace uipc::backend::cuda
