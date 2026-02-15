#include "knockoff_cr/backend.hpp"

PYBIND11_MODULE(knockoff_cr_cpp, m) {
    m.doc() = "Knockoff Clash Royale C++ backend (PPO-oriented)";

    py::class_<knockoff_cr::ClashEnv>(m, "ClashEnv")
        .def(py::init<int, double, int>(), py::arg("tick_hz") = 10, py::arg("max_sim_seconds") = 120.0, py::arg("seed") = 1)
        .def("spec", &knockoff_cr::ClashEnv::spec)
        .def("reset", &knockoff_cr::ClashEnv::reset, py::arg("seed") = -1, py::arg("options") = py::dict())
        .def("step", &knockoff_cr::ClashEnv::step, py::arg("actions"))
        .def("debug_state", &knockoff_cr::ClashEnv::debug_state);

    py::class_<knockoff_cr::ClashEnvBatch>(m, "ClashEnvBatch")
        .def(
            py::init<int, int, double, int, int>(),
            py::arg("num_envs"),
            py::arg("tick_hz") = 10,
            py::arg("max_sim_seconds") = 120.0,
            py::arg("seed") = 1,
            py::arg("num_threads") = 0)
        .def("spec", &knockoff_cr::ClashEnvBatch::spec)
        .def(
            "reset_many",
            &knockoff_cr::ClashEnvBatch::reset_many,
            py::arg("seeds") = py::none(),
            py::arg("options_per_env") = py::none())
        .def(
            "step_many",
            &knockoff_cr::ClashEnvBatch::step_many,
            py::arg("actions_per_env"))
        .def(
            "step_many_discrete",
            &knockoff_cr::ClashEnvBatch::step_many_discrete,
            py::arg("actions_per_env"))
        .def(
            "step_many_packed",
            &knockoff_cr::ClashEnvBatch::step_many_packed,
            py::arg("actions_per_env"))
        .def("debug_state_many", &knockoff_cr::ClashEnvBatch::debug_state_many);
}
