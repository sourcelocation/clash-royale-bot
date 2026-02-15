#include "knockoff_cr/backend.hpp"

#include <algorithm>
#include <cmath>

namespace knockoff_cr {

int ClashEnv::region_width() const {
    return static_cast<int>(std::ceil(static_cast<double>(kGridW) / static_cast<double>(std::max(1, kPlacementRegionCols))));
}

int ClashEnv::region_height() const {
    return static_cast<int>(std::ceil(static_cast<double>(kGridH) / static_cast<double>(std::max(1, kPlacementRegionRows))));
}

int ClashEnv::placement_region_count() const {
    return std::max(1, kPlacementRegionCols * kPlacementRegionRows);
}

int ClashEnv::placement_cells_per_region() const {
    return std::max(1, region_width() * region_height());
}

void ClashEnv::add_reward(int team, double delta, const std::string& term) {
    if (team < 0 || team >= kAgentCount) {
        return;
    }
    rewards_[team] += delta;
    reward_terms_[team][term] += delta;
}

std::pair<int, int> ClashEnv::action_to_internal_grid(int team, int action_x, int action_y) const {
    action_x = std::max(0, std::min(kGridW - 1, action_x));
    action_y = std::max(0, std::min(kGridH - 1, action_y));
    if (team == 0) {
        return {kGridW - 1 - action_x, kGridH - 1 - action_y};
    }
    return {action_x, action_y};
}

std::pair<double, double> ClashEnv::internal_to_world(int gx, int gy) const {
    const double x = static_cast<double>(gx) - (static_cast<double>(kGridW) * 0.5 - 0.5);
    const double y = static_cast<double>(gy) - (static_cast<double>(kGridH) * 0.5 - 0.5);
    return {x, y};
}

std::pair<int, int> ClashEnv::world_to_internal_grid(double x, double y) const {
    int gx = static_cast<int>(std::floor(x + (static_cast<double>(kGridW) * 0.5)));
    int gy = static_cast<int>(std::floor(y + (static_cast<double>(kGridH) * 0.5)));
    gx = std::max(0, std::min(kGridW - 1, gx));
    gy = std::max(0, std::min(kGridH - 1, gy));
    return {gx, gy};
}

bool ClashEnv::is_bridge_internal_x(int gx) const {
    const double x = internal_to_world(gx, kRiverTopRow).first;
    return is_bridge_world_x(x);
}

bool ClashEnv::is_bridge_world_x(double x) const {
    const double arena_left = -(kGridW * 0.5);
    const double arena_right = (kGridW * 0.5);
    const double left_start = arena_left + kBridgeSideMarginTiles;
    const double left_end = left_start + kBridgeWidthTiles;
    const double right_end = arena_right - kBridgeSideMarginTiles;
    const double right_start = right_end - kBridgeWidthTiles;
    const bool on_left_bridge = (x >= left_start && x <= left_end);
    const bool on_right_bridge = (x >= right_start && x <= right_end);
    return on_left_bridge || on_right_bridge;
}

double ClashEnv::bridge_left_center_x() const {
    const double arena_left = -(kGridW * 0.5);
    const double left_start = arena_left + kBridgeSideMarginTiles;
    return left_start + (kBridgeWidthTiles * 0.5);
}

double ClashEnv::bridge_right_center_x() const {
    const double arena_right = (kGridW * 0.5);
    const double right_end = arena_right - kBridgeSideMarginTiles;
    return right_end - (kBridgeWidthTiles * 0.5);
}

bool ClashEnv::is_water_internal(int gy) const {
    return gy >= kRiverTopRow && gy <= kRiverBottomRow;
}

bool ClashEnv::is_alive(const Entity& e) const {
    return e.alive && e.hp > 0.0;
}

Entity* ClashEnv::find_entity(int id) {
    for (auto& e : entities_) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

const Entity* ClashEnv::find_entity_const(int id) const {
    for (const auto& e : entities_) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

} // namespace knockoff_cr
