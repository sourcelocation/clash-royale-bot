#include "knockoff_cr/backend.hpp"

#include <algorithm>

namespace knockoff_cr {

ClashEnv::ClashEnv(int tick_hz, double max_sim_seconds, int seed)
    : tick_hz_(std::max(1, tick_hz)),
      dt_(1.0 / static_cast<double>(std::max(1, tick_hz))),
      max_sim_seconds_(max_sim_seconds > 1e-6 ? max_sim_seconds : 120.0),
      ticks_per_step_(std::max(1, tick_hz_)),
      cards_(build_cards()),
      rng_(static_cast<uint32_t>(seed)) {
    reset(-1, py::dict());
}

py::dict ClashEnv::spec() {
    py::dict action_space;
    {
        py::dict d;
        d["size"] = 2;
        d["action_type"] = "discrete";
        action_space["wait"] = d;
    }
    {
        py::dict d;
        d["size"] = kUsableCardCount;
        d["action_type"] = "discrete";
        action_space["card_selection"] = d;
    }
    {
        py::dict d;
        d["size"] = placement_region_count();
        d["action_type"] = "discrete";
        action_space["position_region"] = d;
    }
    {
        py::dict d;
        d["size"] = placement_cells_per_region();
        d["action_type"] = "discrete";
        action_space["position_cell"] = d;
    }

    py::dict obs_schema;
    obs_schema["vector_size"] = static_cast<int>(build_obs_vector(0).size());
    obs_schema["position_masks_cards"] = kUsableCardCount;
    obs_schema["position_masks_per_card"] = kGridW * kGridH;
    obs_schema["spatial_shape"] = py::make_tuple(0, 0, 0);

    py::dict placement_schema;
    placement_schema["grid_width"] = kGridW;
    placement_schema["grid_height"] = kGridH;
    placement_schema["position_count"] = kGridW * kGridH;
    placement_schema["region_cols"] = kPlacementRegionCols;
    placement_schema["region_rows"] = kPlacementRegionRows;
    placement_schema["region_count"] = placement_region_count();
    placement_schema["region_cell_width"] = region_width();
    placement_schema["region_cell_height"] = region_height();
    placement_schema["region_cell_count"] = placement_cells_per_region();

    const int action_mask_size = 2 + kUsableCardCount + placement_region_count() + placement_cells_per_region();

    py::list action_order;
    action_order.append("wait");
    action_order.append("card_selection");
    action_order.append("position_region");
    action_order.append("position_cell");

    py::dict out;
    out["protocol_version"] = "knockoff_env_v1";
    out["obs_version"] = "v1";
    out["schema_version"] = "knockoff_cr_env_v2";
    out["n_agents"] = kAgentCount;
    out["action_space"] = action_space;
    out["action_order"] = action_order;
    out["obs_schema"] = obs_schema;
    out["action_mask_size"] = action_mask_size;
    out["ticks_per_step"] = ticks_per_step_;
    out["placement_schema"] = placement_schema;
    return out;
}

py::dict ClashEnv::reset(int seed_value, py::dict options) {
    if (seed_value >= 0) {
        rng_.seed(static_cast<uint32_t>(seed_value));
    }

    if (options.contains("ticks_per_step")) {
        ticks_per_step_ = std::max(1, py::cast<int>(options["ticks_per_step"]));
    }

    team_controllers_[0] = "external";
    team_controllers_[1] = "external";
    infinite_elixir_[0] = false;
    infinite_elixir_[1] = false;
    if (options.contains("team_controllers")) {
        try {
            py::list ctrls = py::cast<py::list>(options["team_controllers"]);
            if (py::len(ctrls) >= 2) {
                team_controllers_[0] = py::cast<std::string>(ctrls[0]);
                team_controllers_[1] = py::cast<std::string>(ctrls[1]);
            }
        } catch (const std::exception&) {
        }
    }
    if (options.contains("infinite_elixir_teams")) {
        try {
            py::list teams = py::cast<py::list>(options["infinite_elixir_teams"]);
            if (py::len(teams) >= 2) {
                infinite_elixir_[0] = py::cast<bool>(teams[0]);
                infinite_elixir_[1] = py::cast<bool>(teams[1]);
            }
        } catch (const std::exception&) {
        }
    }

    reset_state();

    py::dict info;
    info["reset"] = true;
    info["options"] = options;
    return build_transition(info, py::make_tuple(false, false), 0, ticks_per_step_);
}

py::dict ClashEnv::step(py::list actions) {
    if (py::len(actions) != kAgentCount) {
        py::dict info;
        info["error"] = "invalid_action_count";
        info["expected_actions"] = kAgentCount;
        info["received_actions"] = py::len(actions);
        return build_transition(info, py::make_tuple(false, false), 0, ticks_per_step_);
    }

    std::array<ExternalAction, kAgentCount> parsed_actions;
    for (int team = 0; team < kAgentCount; ++team) {
        const py::dict action = py::cast<py::dict>(actions[team]);
        parsed_actions[static_cast<size_t>(team)] = parse_external_action_dict(action);
    }
    const StepCoreOutcome outcome = step_core(parsed_actions);
    return finalize_step_transition(outcome);
}

StepCoreOutcome ClashEnv::step_core(const std::array<ExternalAction, kAgentCount>& actions) {
    StepCoreOutcome outcome;
    last_action_debug_[0] = ActionDebug{};
    last_action_debug_[1] = ActionDebug{};
    for (int team = 0; team < kAgentCount; ++team) {
        outcome.action_applied[team] = apply_action(team, actions[team]);
    }
    int ticks_elapsed = 0;
    for (; ticks_elapsed < ticks_per_step_; ++ticks_elapsed) {
        simulate_tick();
        if (episode_done_ || episode_truncated_) {
            ++ticks_elapsed;
            break;
        }
    }
    outcome.ticks_elapsed = ticks_elapsed;
    return outcome;
}

py::dict ClashEnv::finalize_step_transition(const StepCoreOutcome& outcome) {
    py::dict info;
    info["action_applied"] = py::make_tuple(outcome.action_applied[0], outcome.action_applied[1]);
    info["ticks_simulated"] = outcome.ticks_elapsed;
    info["ticks_requested"] = ticks_per_step_;
    if ((episode_done_ || episode_truncated_) && outcome.ticks_elapsed < ticks_per_step_) {
        info["terminated_early"] = true;
    }
    return build_transition(
        info,
        py::make_tuple(outcome.action_applied[0], outcome.action_applied[1]),
        outcome.ticks_elapsed,
        ticks_per_step_);
}

} // namespace knockoff_cr
