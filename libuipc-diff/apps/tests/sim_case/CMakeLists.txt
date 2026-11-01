file(GLOB SOURCE "*.cpp" "*.h")
uipc_add_test(sim_case ${SOURCE})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCE})