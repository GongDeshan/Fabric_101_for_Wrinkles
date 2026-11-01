file(GLOB SOURCE "*.cpp" "*.h")
uipc_add_test(diff_sim_case ${SOURCE})
add_subdirectory(diff_sim_solver)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCE})