#include <backends/common/feature_collection.h>
#include <backends/common/sim_system.h>
namespace uipc::backend
{
void FeatureCollection::insert(std::string_view name, S<core::IFeature> feature)
{
    m_features.insert(std::string{name}, feature);
    m_system.m_features.push_back(feature.get());  // record the feature
}
}  // namespace uipc::backend
