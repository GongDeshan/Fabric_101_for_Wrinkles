#pragma once
#include <sim_system.h>
#include <contact_system/global_diff_contact_manager.h>

namespace uipc::backend::cuda
{
class ContactDiffParmReceiver : public SimSystem
{
    friend class GlobalDiffContactManager;

  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    using ClassifyInfo = GlobalDiffContactManager::ClassifyInfo;
    using ClassifiedDiffParmInfo = GlobalDiffContactManager::ClassifiedContactDiffParmInfo;

  protected:
    virtual void do_report(ClassifyInfo& info)            = 0;
    virtual void do_receive(ClassifiedDiffParmInfo& info) = 0;
    virtual void do_build(BuildInfo& info)                = 0;
    virtual void do_init()                                = 0;

  private:
    virtual void do_build() final override;
    void report(ClassifyInfo& info);  // only called by GlobalDiffContactManager
    void receive(ClassifiedDiffParmInfo& info);  // only called by GlobalDiffContactManager
};
}  // namespace uipc::backend::cuda
