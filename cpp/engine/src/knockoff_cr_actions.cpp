#include "knockoff_cr/backend.hpp"
#include <algorithm>
#include <functional>
namespace knockoff_cr {
namespace {
struct DecodedAction {
    int wait = 0;
    int card_selection = -1;
    int position_region = 0;
    int position_cell = 0;
    int action_x = 0;
    int action_y = 0;
    double wx = 0.0;
    double wy = 0.0;
};
DecodedAction resolve_action_for_team(
    int team,
    const ExternalAction& raw,
    int rw,
    int rh,
    int region_count,
    const std::function<std::pair<int, int>(int, int, int)>& to_internal,
    const std::function<std::pair<double, double>(int, int)>& to_world) {
    DecodedAction out;
    out.wait = raw.wait;
    out.card_selection = raw.card_selection;
    out.position_region = raw.position_region;
    out.position_cell = raw.position_cell;
    const int region_idx = std::max(0, std::min(region_count - 1, out.position_region));
    const int region_x = region_idx % kPlacementRegionCols;
    const int region_y = region_idx / kPlacementRegionCols;
    const int max_cell = std::max(1, rw * rh) - 1;
    const int local_idx = std::max(0, std::min(max_cell, out.position_cell));
    const int local_x = local_idx % rw;
    const int local_y = local_idx / rw;
    out.action_x = std::max(0, std::min(kGridW - 1, region_x * rw + local_x));
    out.action_y = std::max(0, std::min(kGridH - 1, region_y * rh + local_y));
    const auto [ix, iy] = to_internal(team, out.action_x, out.action_y);
    const auto [wx, wy] = to_world(ix, iy);
    out.wx = wx;
    out.wy = wy;
    return out;
}
ExternalAction parse_external_action(const py::dict& action) {
    auto to_int = [&](const char* key, int fallback) {
        if (action.contains(key)) {
            try {
                return py::cast<int>(action[key]);
            } catch (const std::exception&) {
                return fallback;
            }
        }
        return fallback;
    };
    ExternalAction out;
    out.wait = to_int("wait", 0);
    out.card_selection = to_int("card_selection", -1);
    out.position_region = to_int("position_region", 0);
    out.position_cell = to_int("position_cell", 0);
    return out;
}
} // namespace
ExternalAction parse_external_action_dict(const py::dict& action) {
    return parse_external_action(action);
}
bool ClashEnv::apply_action(int team, const py::dict& action) {
    return apply_action(team, parse_external_action_dict(action));
}
bool ClashEnv::apply_action(int team, const ExternalAction& raw_action) {
    ActionDebug dbg;
    dbg.valid = true;
    dbg.team = team;
    const DecodedAction decoded = resolve_action_for_team(
        team,
        raw_action,
        region_width(),
        region_height(),
        placement_region_count(),
        [&](int t, int ax, int ay) { return action_to_internal_grid(t, ax, ay); },
        [&](int ix, int iy) { return internal_to_world(ix, iy); });
    dbg.wait = decoded.wait;
    dbg.card_selection = decoded.card_selection;
    dbg.position_region = decoded.position_region;
    dbg.position_cell = decoded.position_cell;
    dbg.grid_x = decoded.action_x;
    dbg.grid_y = decoded.action_y;
    dbg.world_x = decoded.wx;
    dbg.world_y = decoded.wy;
    action_stats_[team].decisions += 1;
    auto reject = [&](const char* reason, bool is_wait) {
        dbg.applied = false;
        dbg.reason = reason;
        if (is_wait) {
            action_stats_[team].wait += 1;
        } else {
            action_stats_[team].rejections[dbg.reason] += 1;
        }
        last_action_debug_[team] = dbg;
        return false;
    };
    if (team_controllers_[team] != "external") {
        return reject("non_external_controller", false);
    }
    if (episode_done_ || episode_truncated_) {
        return reject("episode_done", false);
    }
    if (decoded.wait == 1) {
        return reject("wait", true);
    }
    PlayResult result = play_card_with_reason(team, decoded.card_selection, decoded.action_x, decoded.action_y);
    dbg.applied = result.applied;
    dbg.reason = result.reason;
    dbg.queued = result.applied;
    dbg.spawn_id = result.spawn_id;
    if (result.applied) {
        action_stats_[team].applied += 1;
    } else {
        action_stats_[team].rejections[result.reason] += 1;
    }
    last_action_debug_[team] = dbg;
    return result.applied;
}
PlayResult ClashEnv::play_card_with_reason(int team, int card_id, int action_x, int action_y) {
    PlayResult out;
    if (card_id < 0 || card_id >= kUsableCardCount) {
        out.reason = "invalid_card";
        return out;
    }
    if (action_x < 0 || action_x >= kGridW || action_y < 0 || action_y >= kGridH) {
        out.reason = "out_of_bounds";
        return out;
    }
    const CardDef& card = cards_[card_id];
    if (card.cost > elixir_[team]) {
        out.reason = "insufficient_elixir";
        return out;
    }
    if (!card_in_hand(team, card_id)) {
        out.reason = "card_not_in_hand";
        return out;
    }
    if (!legal_placement_for_card(team, card, action_x, action_y)) {
        out.reason = "illegal_position";
        return out;
    }
    const auto hand = hand_for_team(team);
    const int hog_cost = cards_[kCardHog].cost;
    const bool should_penalize_cheap_cycle_play = (
        std::find(hand.begin(), hand.end(), kCardHog) != hand.end() &&
        elixir_[team] < hog_cost &&
        card_id != kCardHog &&
        card.cost <= kCheapCardMaxCost
    );
    elixir_[team] -= static_cast<double>(card.cost);
    elixir_[team] = clampd(elixir_[team], 0.0, max_elixir_);
    cycle_deck_after_play(team, card_id);
    if (card.type == CARD_BUILDING) {
        if (action_y <= kBuildingBacklineMaxRow) {
            const int depth = (kBuildingBacklineMaxRow - action_y + 1);
            const double penalty = static_cast<double>(depth) * kBuildingBacklinePenaltyBase;
            add_reward(team, -penalty, "building_backline");
        }
    }
    if (should_penalize_cheap_cycle_play) {
        add_reward(team, -kCheapCyclePenalty, "cheap_cycle_play");
    }
    const auto [ix, iy] = action_to_internal_grid(team, action_x, action_y);
    const auto [wx, wy] = internal_to_world(ix, iy);
    PendingSpawn p;
    p.spawn_id = next_spawn_id_++;
    p.team = team;
    p.card_id = card_id;
    p.x = wx;
    p.y = wy;
    p.spawn_at_s = queue_time_s_ + kSpawnMinDelayS;
    p.active_at_s = -1.0;
    p.state = "queued";
    pending_spawns_.push_back(p);
    out.applied = true;
    out.reason = "queued";
    out.spawn_id = p.spawn_id;
    return out;
}
} // namespace knockoff_cr
