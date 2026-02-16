#include "knockoff_cr/backend.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
namespace knockoff_cr {
namespace {
template <typename T>
void erase_indices_desc(std::vector<T>& vec, std::vector<int>& indices) {
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        const int idx = indices[i];
        if (idx >= 0 && idx < static_cast<int>(vec.size())) {
            vec.erase(vec.begin() + idx);
        }
    }
}
template <typename OnHit>
void tick_attack_timers(Entity& e, double dt, OnHit&& on_hit) {
    if (e.attack_windup_rem > 1e-9) {
        e.attack_windup_rem = std::max(0.0, e.attack_windup_rem - dt);
        if (e.attack_windup_rem <= 1e-9) {
            on_hit();
        }
    }
    if (e.attack_recover_rem > 1e-9) {
        e.attack_recover_rem = std::max(0.0, e.attack_recover_rem - dt);
    }
}
template <typename IsValidLock, typename ShouldDrop, typename Acquire>
Entity* refresh_lock(Entity& e, Entity* lock, IsValidLock&& is_valid_lock, ShouldDrop&& should_drop, Acquire&& acquire) {
    if (!lock || !is_valid_lock(*lock)) {
        lock = nullptr;
    } else if (should_drop(*lock)) {
        lock = nullptr;
    }
    if (!lock) {
        lock = acquire();
    }
    e.lock_target_id = lock ? lock->id : -1;
    return lock;
}

constexpr double kDiagStepCost = 1.41421356237;
constexpr double kLineSampleStepTiles = 0.25;
constexpr double kLogLengthTiles = 5.0;
} // namespace
bool ClashEnv::can_target(const Entity& attacker, const Entity& target) const {
    if (!is_alive(target) || attacker.team == target.team) {
        return false;
    }
    if (attacker.target_type == TARGET_BUILDINGS) {
        return target.kind == ENTITY_BUILDING || target.kind == ENTITY_TOWER;
    }
    return target.kind == ENTITY_TROOP || target.kind == ENTITY_BUILDING || target.kind == ENTITY_TOWER;
}
Entity* ClashEnv::nearest_target_within(const Entity& attacker, double max_range) {
    Entity* best = nullptr;
    double best_d2 = std::numeric_limits<double>::infinity();
    const double max_d2 = max_range * max_range;
    for (auto& other : entities_) {
        if (!can_target(attacker, other)) {
            continue;
        }
        const double d2 = dist2(attacker.x, attacker.y, other.x, other.y);
        if (d2 <= max_d2 && d2 < best_d2) {
            best_d2 = d2;
            best = &other;
        }
    }
    return best;
}
Entity* ClashEnv::nearest_princess_tower(const Entity& seeker, int enemy_team) {
    Entity* best = nullptr;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (auto& e : entities_) {
        if (!is_alive(e) || e.kind != ENTITY_TOWER || e.team != enemy_team || e.king_tower) {
            continue;
        }
        const double d2 = dist2(seeker.x, seeker.y, e.x, e.y);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = &e;
        }
    }
    return best;
}
bool ClashEnv::target_in_attack_range(const Entity& attacker, const Entity& target) const {
    const double reach = std::max(0.0, attacker.attack_range) + std::max(0.0, attacker.radius) + std::max(0.0, target.radius);
    return dist2(attacker.x, attacker.y, target.x, target.y) <= (reach * reach);
}
std::pair<double, double> ClashEnv::movement_target(const Entity& e) {
    if (const Entity* lock = find_entity_const(e.lock_target_id)) {
        if (is_alive(*lock)) {
            return {lock->x, lock->y};
        }
    }
    Entity* fallback = nullptr;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (auto& other : entities_) {
        if (!is_alive(other) || other.kind != ENTITY_TOWER || other.team == e.team || other.king_tower) {
            continue;
        }
        const double d2 = dist2(e.x, e.y, other.x, other.y);
        if (d2 < best_d2) {
            best_d2 = d2;
            fallback = &other;
        }
    }
    if (fallback) {
        return {fallback->x, fallback->y};
    }
    return {0.0, (e.team == 0) ? -(kGridH * 0.5) : (kGridH * 0.5)};
}
std::pair<double, double> ClashEnv::bridge_waypoint_if_needed(const Entity& e, double tx, double ty) {
    if (e.card_id == kCardHog || e.kind != ENTITY_TROOP) {
        return {tx, ty};
    }
    const auto [egi_x, egi_y] = world_to_internal_grid(e.x, e.y);
    const auto [tgi_x, tgi_y] = world_to_internal_grid(tx, ty);
    (void)egi_x;
    (void)tgi_x;
    const bool need_cross = (e.team == 0) ? (egi_y > kRiverBottomRow && tgi_y < kRiverTopRow)
                                           : (egi_y < kRiverTopRow && tgi_y > kRiverBottomRow);
    if (!need_cross) {
        return {tx, ty};
    }
    const double bridge0 = bridge_left_center_x();
    const double bridge1 = bridge_right_center_x();
    const double bx = (std::abs(e.x - bridge0) < std::abs(e.x - bridge1)) ? bridge0 : bridge1;
    const double by = (e.team == 0) ? internal_to_world(0, kRiverBottomRow).second : internal_to_world(0, kRiverTopRow).second;
    return {bx, by};
}

bool ClashEnv::cell_walkable_for(const Entity& e, int gx, int gy) const {
    if (gx < 0 || gx >= kGridW || gy < 0 || gy >= kGridH) {
        return false;
    }
    if (e.card_id == kCardHog) {
        return true;
    }
    if (is_water_internal(gy) && !is_bridge_internal_x(gx)) {
        return false;
    }
    return true;
}

bool ClashEnv::line_walkable_for(const Entity& e, double x0, double y0, double x1, double y1) const {
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len <= 1e-9) {
        const auto [gx, gy] = world_to_internal_grid(x0, y0);
        return cell_walkable_for(e, gx, gy);
    }
    const int samples = std::max(1, static_cast<int>(std::ceil(len / kLineSampleStepTiles)));
    for (int i = 0; i <= samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples);
        const double sx = x0 + dx * t;
        const double sy = y0 + dy * t;
        const auto [gx, gy] = world_to_internal_grid(sx, sy);
        if (!cell_walkable_for(e, gx, gy)) {
            return false;
        }
    }
    return true;
}

std::pair<double, double> ClashEnv::next_path_waypoint(const Entity& e, double tx, double ty) const {
    if (line_walkable_for(e, e.x, e.y, tx, ty)) {
        return {tx, ty};
    }

    const auto [sx, sy] = world_to_internal_grid(e.x, e.y);
    const auto [gx, gy] = world_to_internal_grid(tx, ty);
    if (sx == gx && sy == gy) {
        return {tx, ty};
    }

    const int total = kGridW * kGridH;
    const auto to_idx = [](int x, int y) { return y * kGridW + x; };
    const auto from_idx = [](int idx) { return std::pair<int, int>{idx % kGridW, idx / kGridW}; };
    const int start = to_idx(sx, sy);
    const int goal = to_idx(gx, gy);

    std::array<double, kGridW * kGridH> gscore;
    std::array<int, kGridW * kGridH> parent;
    std::array<uint8_t, kGridW * kGridH> state;
    gscore.fill(std::numeric_limits<double>::infinity());
    parent.fill(-1);
    state.fill(0);

    auto heuristic = [&](int x, int y) {
        const double dx = std::abs(static_cast<double>(x - gx));
        const double dy = std::abs(static_cast<double>(y - gy));
        const double lo = std::min(dx, dy);
        const double hi = std::max(dx, dy);
        return lo * kDiagStepCost + (hi - lo);
    };

    std::vector<int> open;
    open.reserve(128);
    open.push_back(start);
    state[start] = 1;
    gscore[start] = 0.0;

    while (!open.empty()) {
        int best_i = 0;
        double best_f = std::numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(open.size()); ++i) {
            const int idx = open[i];
            const auto [cx, cy] = from_idx(idx);
            const double f = gscore[idx] + heuristic(cx, cy);
            if (f < best_f) {
                best_f = f;
                best_i = i;
            }
        }
        const int current = open[best_i];
        open[best_i] = open.back();
        open.pop_back();
        state[current] = 2;

        if (current == goal) {
            break;
        }

        const auto [cx, cy] = from_idx(current);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int nx = cx + dx;
                const int ny = cy + dy;
                if (!cell_walkable_for(e, nx, ny)) {
                    continue;
                }
                const bool diagonal = (dx != 0 && dy != 0);
                if (diagonal) {
                    if (!cell_walkable_for(e, cx + dx, cy) || !cell_walkable_for(e, cx, cy + dy)) {
                        continue;
                    }
                }
                const int nidx = to_idx(nx, ny);
                if (state[nidx] == 2) {
                    continue;
                }
                const double step_cost = diagonal ? kDiagStepCost : 1.0;
                const double cand_g = gscore[current] + step_cost;
                if (cand_g + 1e-9 >= gscore[nidx]) {
                    continue;
                }
                gscore[nidx] = cand_g;
                parent[nidx] = current;
                if (state[nidx] != 1) {
                    state[nidx] = 1;
                    open.push_back(nidx);
                }
            }
        }
    }

    if (parent[goal] == -1) {
        return {tx, ty};
    }

    int step_idx = goal;
    while (parent[step_idx] != -1 && parent[step_idx] != start) {
        step_idx = parent[step_idx];
    }
    const auto [wx_cell, wy_cell] = from_idx(step_idx);
    const auto [wx, wy] = internal_to_world(wx_cell, wy_cell);
    return {wx, wy};
}

void ClashEnv::apply_damage(Entity& target, int dmg, int source_team, int source_card, bool from_spell) {
    if (!is_alive(target) || dmg <= 0) {
        return;
    }
    target.hp -= static_cast<double>(dmg);
    if (target.kind == ENTITY_TOWER) {
        time_since_tower_damaged_ = 0.0;
    }
    if (target.hp > 0.0) {
        return;
    }
    target.alive = false;
    if (target.kind == ENTITY_TOWER) {
        if (target.team == 0) {
            add_reward(0, -kTerminalReward, "tower_destroyed");
            add_reward(1, kTerminalReward, "tower_kill");
        } else {
            add_reward(0, kTerminalReward, "tower_kill");
            add_reward(1, -kTerminalReward, "tower_destroyed");
        }
        episode_done_ = true;
        reset_requested_ = true;
    }
    if (target.card_id == kCardIceGolem) {
        const auto& card = cards_[kCardIceGolem];
        for (auto& other : entities_) {
            if (!is_alive(other) || other.team == target.team) {
                continue;
            }
            if (dist2(target.x, target.y, other.x, other.y) <= (1.5 * 1.5)) {
                other.hp -= static_cast<double>(card.damage);
                other.stun_rem = std::max(other.stun_rem, 2.0);
                if (other.hp <= 0.0) {
                    other.alive = false;
                }
            }
        }
    }
    (void)source_team;
    (void)source_card;
    (void)from_spell;
}
void ClashEnv::resolve_spell_reward(int caster_team, int hit_count) {
    (void)caster_team;
    (void)hit_count;
}
void ClashEnv::process_fireballs() {
    std::vector<int> remove;
    remove.reserve(fireballs_.size());
    for (int i = 0; i < static_cast<int>(fireballs_.size()); ++i) {
        auto& f = fireballs_[i];
        if (queue_time_s_ < f.detonate_at_s) {
            continue;
        }
        int hit_count = 0;
        const auto& card = cards_[kCardFireball];
        const double r2 = card.attack_range * card.attack_range;
        for (auto& e : entities_) {
            if (!is_alive(e) || e.team == f.team) {
                continue;
            }
            if (dist2(f.x, f.y, e.x, e.y) <= r2) {
                ++hit_count;
                const int dmg = (e.kind == ENTITY_TOWER) ? card.tower_damage : card.damage;
                apply_damage(e, dmg, f.team, kCardFireball, true);
                e.stun_rem = std::max(e.stun_rem, 0.4);
            }
        }
        resolve_spell_reward(f.team, hit_count);
        remove.push_back(i);
    }
    erase_indices_desc(fireballs_, remove);
}
void ClashEnv::process_logs() {
    std::vector<int> remove;
    remove.reserve(logs_.size());
    for (int i = 0; i < static_cast<int>(logs_.size()); ++i) {
        auto& s = logs_[i];
        s.time_left_s -= dt_;
        const double prev_y = s.y;
        s.y += cards_[kCardLog].speed * s.dir_y * dt_;
        const double half_width = cards_[kCardLog].attack_range;
        const double half_len = 0.5 * kLogLengthTiles;
        const double min_x = s.x - half_len;
        const double max_x = s.x + half_len;
        const double min_y = std::min(prev_y, s.y) - half_width;
        const double max_y = std::max(prev_y, s.y) + half_width;
        for (auto& e : entities_) {
            if (!is_alive(e) || e.team == s.team) {
                continue;
            }
            if (s.hit_entity_ids.count(e.id) > 0) {
                continue;
            }
            const bool in_x = (e.x >= min_x && e.x <= max_x);
            const bool in_y = (e.y >= min_y && e.y <= max_y);
            if (in_x && in_y) {
                s.hit_entity_ids.insert(e.id);
                ++s.hit_count;
                const int dmg = (e.kind == ENTITY_TOWER) ? cards_[kCardLog].tower_damage : cards_[kCardLog].damage;
                apply_damage(e, dmg, s.team, kCardLog, true);
                e.stun_rem = std::max(e.stun_rem, 0.4);
                if (e.kind == ENTITY_TROOP) {
                    e.attack_windup_rem = 0.0;
                    e.attack_recover_rem = 0.0;
                }
            }
        }
        if (s.time_left_s <= 0.0) {
            resolve_spell_reward(s.team, s.hit_count);
            remove.push_back(i);
        }
    }
    erase_indices_desc(logs_, remove);
}
void ClashEnv::update_entities() {
    for (auto& e : entities_) {
        if (!is_alive(e)) {
            continue;
        }
        e.stun_rem = std::max(0.0, e.stun_rem - dt_);
        e.deployment_lock_rem = std::max(0.0, e.deployment_lock_rem - dt_);
        if (e.kind == ENTITY_BUILDING && e.decay_time > 1e-9) {
            e.hp -= (e.max_hp / e.decay_time) * dt_;
            if (e.hp <= 0.0) {
                e.alive = false;
                continue;
            }
        }
        if (e.kind == ENTITY_TOWER) {
            update_tower(e);
        } else if (e.kind == ENTITY_BUILDING) {
            update_stationary_attacker(e);
        } else {
            update_troop(e);
        }
    }
}
void ClashEnv::update_tower(Entity& e) {
    if (!e.tower_active || e.stun_rem > 1e-9 || e.deployment_lock_rem > 1e-9) {
        return;
    }
    e.tower_cooldown_rem = std::max(0.0, e.tower_cooldown_rem - dt_);
    Entity* lock = find_entity(e.lock_target_id);
    if (!lock || !can_target(e, *lock) || !target_in_attack_range(e, *lock)) {
        lock = nearest_target_within(e, kTowerAttackRangeTiles);
        e.lock_target_id = lock ? lock->id : -1;
    }
    if (!lock) {
        return;
    }
    if (e.tower_cooldown_rem <= 1e-9) {
        const int dmg = (lock->kind == ENTITY_TOWER) ? e.tower_damage : e.damage;
        apply_damage(*lock, dmg, e.team, -1, false);
        e.tower_cooldown_rem = kTowerAttackCooldownS;
    }
}
void ClashEnv::begin_attack_if_possible(Entity& e, Entity& target) {
    if (e.stun_rem > 1e-9 || e.deployment_lock_rem > 1e-9) {
        return;
    }
    if (e.attack_windup_rem > 1e-9 || e.attack_recover_rem > 1e-9) {
        return;
    }
    if (target_in_attack_range(e, target)) {
        e.attack_windup_rem = std::max(0.0, e.attack_swing);
    }
}
void ClashEnv::resolve_attack_hit(Entity& e) {
    Entity* target = find_entity(e.lock_target_id);
    if (!target || !can_target(e, *target) || !target_in_attack_range(e, *target)) {
        return;
    }
    const int dmg = (target->kind == ENTITY_TOWER) ? e.tower_damage : e.damage;
    apply_damage(*target, dmg, e.team, e.card_id, false);
    if (e.card_id == kCardIceSpirit) {
        const double aoe_r2 = 1.5 * 1.5;
        for (auto& other : entities_) {
            if (!is_alive(other) || other.team == e.team || other.id == target->id) {
                continue;
            }
            if (dist2(target->x, target->y, other.x, other.y) <= aoe_r2) {
                apply_damage(other, cards_[kCardIceSpirit].damage, e.team, kCardIceSpirit, false);
                other.stun_rem = std::max(other.stun_rem, 1.2);
            }
        }
        target->stun_rem = std::max(target->stun_rem, 1.2);
        e.alive = false;
    }
    e.attack_recover_rem = std::max(0.0, e.attack_end);
}
void ClashEnv::update_stationary_attacker(Entity& e) {
    tick_attack_timers(e, dt_, [&]() { resolve_attack_hit(e); });
    Entity* lock = refresh_lock(
        e,
        find_entity(e.lock_target_id),
        [&](const Entity& target) { return can_target(e, target) && target_in_attack_range(e, target); },
        [&](const Entity&) { return false; },
        [&]() { return nearest_target_within(e, kSenseRadiusTiles); });
    if (lock) {
        begin_attack_if_possible(e, *lock);
    }
}
void ClashEnv::update_troop(Entity& e) {
    tick_attack_timers(e, dt_, [&]() { resolve_attack_hit(e); });
    const double sense_r2 = kSenseRadiusTiles * kSenseRadiusTiles;
    const bool attack_locked = (e.attack_windup_rem > 1e-9 || e.attack_recover_rem > 1e-9);
    Entity* lock = find_entity(e.lock_target_id);
    auto acquire_nearest = [&]() {
        Entity* nearest = nearest_target_within(e, kSenseRadiusTiles);
        if (nearest) {
            return nearest;
        }
        return nearest_princess_tower(e, 1 - e.team);
    };

    if (attack_locked) {
        if (!lock || !can_target(e, *lock)) {
            lock = acquire_nearest();
        }
    } else {
        // While approaching, always re-evaluate and pick the closest valid target.
        lock = acquire_nearest();
    }
    e.lock_target_id = lock ? lock->id : -1;
    if (lock) {
        begin_attack_if_possible(e, *lock);
    }
    if (e.stun_rem > 1e-9 || e.deployment_lock_rem > 1e-9 || e.attack_windup_rem > 1e-9 || e.attack_recover_rem > 1e-9) {
        return;
    }
    const auto [tx0, ty0] = movement_target(e);
    auto [tx, ty] = next_path_waypoint(e, tx0, ty0);
    const double dx = tx - e.x;
    const double dy = ty - e.y;
    const double len = std::sqrt(dx * dx + dy * dy);
    if (len <= 1e-9) {
        return;
    }
    const double step = e.speed * dt_;
    const double ux = dx / len;
    const double uy = dy / len;
    double nx = e.x + ux * step;
    double ny = e.y + uy * step;
    e.x = clampd(nx, -(kGridW / 2.0), (kGridW / 2.0));
    e.y = clampd(ny, -(kGridH / 2.0), (kGridH / 2.0));
}
void ClashEnv::process_push_collision() {
    for (auto& a : entities_) {
        if (!is_alive(a) || a.kind != ENTITY_TROOP) {
            continue;
        }
        for (const auto& b : entities_) {
            if (!is_alive(b) || a.id == b.id) {
                continue;
            }
            const double rsum = a.radius + b.radius;
            const double d2 = dist2(a.x, a.y, b.x, b.y);
            if (d2 <= 1e-12 || d2 >= rsum * rsum) {
                continue;
            }
            const double d = std::sqrt(d2);
            const double overlap = (rsum - d) * 0.5;
            const double nx = (a.x - b.x) / d;
            const double ny = (a.y - b.y) / d;
            a.x += nx * overlap;
            a.y += ny * overlap;
        }
        a.x = clampd(a.x, -(kGridW / 2.0), (kGridW / 2.0));
        a.y = clampd(a.y, -(kGridH / 2.0), (kGridH / 2.0));
    }
}
void ClashEnv::cleanup_dead_entities() {
    entities_.erase(
        std::remove_if(
            entities_.begin(),
            entities_.end(),
            [&](const Entity& e) {
                if (is_alive(e)) {
                    return false;
                }
                return e.kind != ENTITY_TOWER;
            }),
        entities_.end());
    for (auto& e : entities_) {
        if (!is_alive(e)) {
            e.lock_target_id = -1;
        }
        if (const Entity* lock = find_entity_const(e.lock_target_id)) {
            if (!is_alive(*lock)) {
                e.lock_target_id = -1;
            }
        } else {
            e.lock_target_id = -1;
        }
    }
}
} // namespace knockoff_cr
