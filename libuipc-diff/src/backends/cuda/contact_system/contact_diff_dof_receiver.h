#pragma once
#include <sim_system.h>
#include <contact_system/global_diff_contact_manager.h>

namespace uipc::backend::cuda
{
class ContactDiffDofReceiver : public SimSystem
{
    friend class GlobalDiffContactManager;

  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    using ClassifyInfo = GlobalDiffContactManager::ClassifyInfo;
    using ClassifiedDiffDofInfo = GlobalDiffContactManager::ClassifiedContactDiffDofInfo;

  protected:
    virtual void do_report(ClassifyInfo& info)           = 0;
    virtual void do_receive(ClassifiedDiffDofInfo& info) = 0;
    virtual void do_build(BuildInfo& info)               = 0;
    virtual void do_init()                               = 0;

  private:
    virtual void do_build() final override;
};
}  // namespace uipc::backend::cuda
