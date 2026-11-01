#pragma once
#include <uipc/core/feature_collection.h>

namespace uipc::backend
{
class SimSystem;
class FeatureCollection
{
  public:
    FeatureCollection(SimSystem& system, core::FeatureCollection& features) noexcept
        : m_features{features}
        , m_system{system}
    {
    }

    template <std::derived_from<core::IFeature> T>
    S<T> find(std::string_view name = T::FeatureName) const
    {
        return m_features.template find<T>(name);
    }

    template <std::derived_from<core::IFeature> T>
    void insert(S<T> feature)
    {
        insert(T::FeatureName, feature);
    }

    void insert(std::string_view name, S<core::IFeature> feature);

  private:
    core::FeatureCollection& m_features;
    SimSystem&               m_system;
};
}  // namespace uipc::backend
