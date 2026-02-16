#include "knockoff_cr/backend.hpp"
#include <algorithm>
#include <cmath>
namespace knockoff_cr {
namespace {
const char* entity_kind_name(EntityKind kind) {
    switch (kind) {
        case ENTITY_TROOP:
            return "troop";
        case ENTITY_BUILDING:
            return "building";
        case ENTITY_TOWER:
            return "tower";
        default:
            return "unknown";
    }
}
} // namespace
void ClashEnv::reset_state() {
    entities_.clear();
    pending_spawns_.clear();
    fireballs_.clear();
    logs_.clear();
    next_entity_id_ = 1;
    next_spawn_id_ = 1;
    rewards_ = {0.0, 0.0};
    reward_terms_[0].clear();
    reward_terms_[1].clear();
    action_stats_[0] = TeamStats{};
    action_stats_[1] = TeamStats{};
    last_action_debug_[0] = ActionDebug{};
    last_action_debug_[1] = ActionDebug{};
    sim_time_s_ = 0.0;
    queue_accum_s_ = 0.0;
    queue_time_s_ = 0.0;
    time_since_tower_damaged_ = 0.0;
    time_since_tower_reward_ = 0.0;
    episode_done_ = false;
    episode_truncated_ = false;
    reset_requested_ = false;
    elixir_[0] = 8.0;
    elixir_[1] = 8.0;
    if (infinite_elixir_[0]) {
        elixir_[0] = max_elixir_;
    }
    if (infinite_elixir_[1]) {
        elixir_[1] = max_elixir_;
    }
    for (int t = 0; t < kAgentCount; ++t) {
        decks_[t] = {0, 1, 2, 3, 4, 5, 6, 7};
        std::shuffle(decks_[t].begin(), decks_[t].end(), rng_);
    }
    add_tower(/*team=*/1, /*king=*/true, /*active=*/false, /*gx=*/8, /*gy=*/2, /*hp=*/7030.0);
    add_tower(/*team=*/1, /*king=*/false, /*active=*/true, /*gx=*/5, /*gy=*/4, /*hp=*/4420.0);
    add_tower(/*team=*/1, /*king=*/false, /*active=*/true, /*gx=*/12, /*gy=*/4, /*hp=*/4420.0);
    add_tower(/*team=*/0, /*king=*/true, /*active=*/false, /*gx=*/8, /*gy=*/29, /*hp=*/7030.0);
    add_tower(/*team=*/0, /*king=*/false, /*active=*/true, /*gx=*/5, /*gy=*/27, /*hp=*/4420.0);
    add_tower(/*team=*/0, /*king=*/false, /*active=*/true, /*gx=*/12, /*gy=*/27, /*hp=*/4420.0);
    initial_tower_health_[0] = sum_visible_tower_health(0);
    initial_tower_health_[1] = sum_visible_tower_health(1);
    last_tower_health_ = initial_tower_health_;
}
void ClashEnv::add_tower(int team, bool king, bool active, int gx, int gy, double hp) {
    Entity e;
    e.id = next_entity_id_++;
    e.kind = ENTITY_TOWER;
    e.team = team;
    e.card_id = -1;
    e.king_tower = king;
    e.tower_active = active;
    auto [x, y] = internal_to_world(gx, gy);
    if (king) {
        x = 0.0;
    } else {
        x = (x < 0.0) ? bridge_left_center_x() : bridge_right_center_x();
    }
    y += (team == 1) ? 0.5 : -0.5;
    e.x = x;
    e.y = y;
    e.hp = hp;
    e.max_hp = hp;
    e.damage = kTowerDamage;
    e.tower_damage = kTowerDamage;
    e.attack_range = kTowerAttackRangeTiles;
    e.attack_swing = 0.0;
    e.attack_end = kTowerAttackCooldownS;
    e.target_type = TARGET_BOTH;
    e.decay_time = 0.0;
    e.speed = 0.0;
    e.radius = 1.0;
    e.tower_cooldown_rem = 0.0;
    entities_.push_back(e);
}
std::vector<int> ClashEnv::hand_for_team(int team) const {
    std::vector<int> hand;
    hand.reserve(kHandSize);
    for (int i = 0; i < kHandSize && i < static_cast<int>(decks_[team].size()); ++i) {
        hand.push_back(decks_[team][i]);
    }
    return hand;
}
bool ClashEnv::team_has_infinite_elixir(int team) const {
    if (team < 0 || team >= kAgentCount) {
        return false;
    }
    return infinite_elixir_[team];
}
bool ClashEnv::card_in_hand(int team, int card_id) const {
    const auto hand = hand_for_team(team);
    return std::find(hand.begin(), hand.end(), card_id) != hand.end();
}
void ClashEnv::cycle_deck_after_play(int team, int card_id) {
    auto& deck = decks_[team];
    const int hand_size = std::min(kHandSize, static_cast<int>(deck.size()));
    auto it = std::find(deck.begin(), deck.begin() + hand_size, card_id);
    if (it == deck.begin() + hand_size) {
        return;
    }
    const int index_in_hand = static_cast<int>(std::distance(deck.begin(), it));
    deck.erase(deck.begin() + index_in_hand);
    deck.push_back(card_id);
    if (static_cast<int>(deck.size()) > 3) {
        const int new_card = deck[3];
        deck.erase(deck.begin() + 3);
        deck.insert(deck.begin() + index_in_hand, new_card);
    }
}
bool ClashEnv::legal_placement_for_card(int team, const CardDef& card, int action_x, int action_y) const {
    if (action_x < 0 || action_x >= kGridW || action_y < 0 || action_y >= kGridH) {
        return false;
    }
    if (card.type == CARD_SPELL) {
        return true;
    }
    if (action_y >= kHalfH) {
        return false;
    }
    const auto [ix, iy] = action_to_internal_grid(team, action_x, action_y);
    if (is_water_internal(iy)) {
        return false;
    }
    (void)ix;
    return true;
}
int ClashEnv::fill_legal_positions(int team, const CardDef& card, std::vector<double>* mask_out) const {
    int legal_count = 0;
    for (int y = 0; y < kGridH; ++y) {
        for (int x = 0; x < kGridW; ++x) {
            const bool legal = legal_placement_for_card(team, card, x, y);
            if (legal) {
                ++legal_count;
            }
            if (mask_out != nullptr) {
                (*mask_out)[y * kGridW + x] = legal ? 1.0 : 0.0;
            }
        }
    }
    return legal_count;
}
std::vector<double> ClashEnv::position_mask_for_card(int team, int card_id) const {
    std::vector<double> out;
    out.assign(kGridW * kGridH, 0.0);
    if (card_id < 0 || card_id >= kUsableCardCount) {
        return out;
    }
    if (!card_in_hand(team, card_id)) {
        return out;
    }
    const CardDef& card = cards_[card_id];
    if (!team_has_infinite_elixir(team) && card.cost > elixir_[team]) {
        return out;
    }
    fill_legal_positions(team, card, &out);
    return out;
}
std::vector<double> ClashEnv::action_mask_flat(int team) const {
    const int region_count = placement_region_count();
    const int cell_count = placement_cells_per_region();
    std::vector<double> out;
    out.reserve(2 + kUsableCardCount + region_count + cell_count);
    out.push_back(0.0);
    out.push_back(1.0);
    bool has_playable = false;
    const auto hand = hand_for_team(team);
    for (int card_id = 0; card_id < kUsableCardCount; ++card_id) {
        const CardDef& card = cards_[card_id];
        const bool in_hand = std::find(hand.begin(), hand.end(), card_id) != hand.end();
        const bool has_elixir = team_has_infinite_elixir(team) || card.cost <= elixir_[team];
        const bool can_play = in_hand && has_elixir && fill_legal_positions(team, card, nullptr) > 0;
        out.push_back(can_play ? 1.0 : 0.0);
        has_playable = has_playable || can_play;
    }
    out[0] = has_playable ? 1.0 : 0.0;
    out.insert(out.end(), region_count, has_playable ? 1.0 : 0.0);
    out.insert(out.end(), cell_count, has_playable ? 1.0 : 0.0);
    return out;
}
int ClashEnv::count_alive_towers(int team) const {
    int count = 0;
    for (const auto& e : entities_) {
        if (!is_alive(e)) {
            continue;
        }
        if (e.kind == ENTITY_TOWER && e.team == team) {
            ++count;
        }
    }
    return count;
}
double ClashEnv::sum_visible_tower_health(int team) const {
    double total = 0.0;
    for (const auto& e : entities_) {
        if (!is_alive(e)) {
            continue;
        }
        if (e.kind == ENTITY_TOWER && e.team == team) {
            total += e.hp;
        }
    }
    return total;
}
int ClashEnv::count_playable_cards(int team) const {
    int count = 0;
    const auto hand = hand_for_team(team);
    for (int card_id : hand) {
        if (card_id < 0 || card_id >= kUsableCardCount) {
            continue;
        }
        const auto& c = cards_[card_id];
        const bool has_elixir = team_has_infinite_elixir(team) || c.cost <= elixir_[team];
        if (has_elixir && fill_legal_positions(team, c, nullptr) > 0) {
            ++count;
        }
    }
    return count;
}
std::vector<float> ClashEnv::build_obs_vector(int for_team) const {
    const bool should_flip = (for_team == 1);
    const int team_self = should_flip ? 1 : 0;
    const int team_enemy = should_flip ? 0 : 1;
    std::vector<float> res;
    res.reserve(512);
    auto append_card_onehot = [&](int card_id) {
        for (int c = 0; c < kUsableCardCount; ++c) {
            res.push_back((card_id == c) ? 1.0f : 0.0f);
        }
    };
    const double self_elixir = should_flip ? elixir_[1] : elixir_[0];
    const double enemy_elixir = should_flip ? elixir_[0] : elixir_[1];
    res.push_back(static_cast<float>(std::min(self_elixir, 10.0) / 10.0));
    res.push_back(static_cast<float>(std::min(enemy_elixir, 10.0) / 10.0));
    res.push_back(static_cast<float>(clampd(time_since_tower_damaged_ / 15.0, 0.0, 1.0)));
    res.push_back(static_cast<float>(static_cast<double>(count_alive_towers(team_self)) / 3.0));
    res.push_back(static_cast<float>(static_cast<double>(count_alive_towers(team_enemy)) / 3.0));
    res.push_back(static_cast<float>(static_cast<double>(count_playable_cards(team_self)) / static_cast<double>(kHandSize)));
    res.push_back(static_cast<float>(static_cast<double>(count_playable_cards(team_enemy)) / static_cast<double>(kHandSize)));
    const auto hand0 = hand_for_team(0);
    const auto hand1 = hand_for_team(1);
    for (int i = 0; i < kHandSize; ++i) {
        const int a = (i < static_cast<int>(hand0.size())) ? hand0[i] : -1;
        const int b = (i < static_cast<int>(hand1.size())) ? hand1[i] : -1;
        const int self_card = should_flip ? b : a;
        const int enemy_card = should_flip ? a : b;
        append_card_onehot(self_card);
        append_card_onehot(enemy_card);
    }
    std::vector<const Entity*> team_a;
    std::vector<const Entity*> team_b;
    team_a.reserve(32);
    team_b.reserve(32);
    for (const auto& e : entities_) {
        if (!is_alive(e)) {
            continue;
        }
        if (e.team == (should_flip ? 1 : 0)) {
            team_a.push_back(&e);
        } else {
            team_b.push_back(&e);
        }
    }
    auto sorter = [should_flip](const Entity* lhs, const Entity* rhs) {
        const double az = should_flip ? -lhs->y : lhs->y;
        const double bz = should_flip ? -rhs->y : rhs->y;
        if (std::abs(az - bz) > 1e-9) {
            return az < bz;
        }
        const double ax = should_flip ? -lhs->x : lhs->x;
        const double bx = should_flip ? -rhs->x : rhs->x;
        if (std::abs(ax - bx) > 1e-9) {
            return ax < bx;
        }
        return lhs->id < rhs->id;
    };
    std::sort(team_a.begin(), team_a.end(), sorter);
    std::sort(team_b.begin(), team_b.end(), sorter);
    auto append_team_entities = [&](const std::vector<const Entity*>& team_entities) {
        for (int i = 0; i < kMaxEntitiesPerTeamObs; ++i) {
            if (i < static_cast<int>(team_entities.size())) {
                const Entity* e = team_entities[i];
                const double x = should_flip ? -e->x : e->x;
                const double y = should_flip ? -e->y : e->y;
                res.push_back(1.0f);
                res.push_back(static_cast<float>((x + static_cast<double>(kGridW) / 2.0) / static_cast<double>(kGridW)));
                res.push_back(static_cast<float>((y + static_cast<double>(kGridH) / 2.0) / static_cast<double>(kGridH)));
                append_card_onehot(e->card_id);
                const double hp_frac = e->max_hp > 1e-9 ? e->hp / e->max_hp : 0.0;
                res.push_back(static_cast<float>(clampd(hp_frac, 0.0, 1.0)));
                const bool is_building = (e->kind == ENTITY_BUILDING || e->kind == ENTITY_TOWER);
                res.push_back(is_building ? 1.0f : 0.0f);
            } else {
                res.push_back(0.0f);
                res.push_back(0.0f);
                res.push_back(0.0f);
                for (int c = 0; c < kUsableCardCount; ++c) {
                    res.push_back(0.0f);
                }
                res.push_back(0.0f);
                res.push_back(0.0f);
            }
        }
    };
    append_team_entities(team_a);
    append_team_entities(team_b);
    for (auto& v : res) {
        if (!is_finite(v)) {
            v = 0.0f;
        }
    }
    return res;
}
std::vector<std::vector<double>> ClashEnv::all_card_position_masks(int team) const {
    std::vector<std::vector<double>> masks;
    masks.reserve(kUsableCardCount);
    for (int card = 0; card < kUsableCardCount; ++card) {
        masks.push_back(position_mask_for_card(team, card));
    }
    return masks;
}
py::dict ClashEnv::build_transition(py::dict info, py::tuple action_applied, int ticks_simulated, int ticks_requested) {
    (void)action_applied;
    (void)ticks_simulated;
    (void)ticks_requested;
    py::list obs;
    py::list action_mask;
    py::list reward;
    py::list reward_terms;
    py::list action_debug;
    py::list action_stats;
    for (int team = 0; team < kAgentCount; ++team) {
        py::dict obs_entry;
        obs_entry["vector"] = build_obs_vector(team);
        obs_entry["position_masks_for_all_cards"] = all_card_position_masks(team);
        obs.append(obs_entry);
        action_mask.append(action_mask_flat(team));
        reward.append(rewards_[team]);
        py::dict term_dict;
        for (const auto& kv : reward_terms_[team]) {
            term_dict[py::str(kv.first)] = kv.second;
        }
        reward_terms.append(term_dict);
        if (last_action_debug_[team].valid) {
            const auto& dbg = last_action_debug_[team];
            py::dict d;
            d["team"] = dbg.team;
            d["wait"] = dbg.wait;
            d["card_selection"] = dbg.card_selection;
            d["position_region"] = dbg.position_region;
            d["position_cell"] = dbg.position_cell;
            d["grid_x"] = dbg.grid_x;
            d["grid_y"] = dbg.grid_y;
            d["world_x"] = dbg.world_x;
            d["world_z"] = dbg.world_y;
            d["applied"] = dbg.applied;
            d["reason"] = dbg.reason;
            d["queued"] = dbg.queued;
            d["spawn_id"] = dbg.spawn_id;
            d["queue_hz"] = dbg.queue_hz;
            d["spawn_min_delay_s"] = dbg.spawn_min_delay_s;
            d["deploy_min_delay_s"] = dbg.deploy_min_delay_s;
            action_debug.append(d);
        } else {
            action_debug.append(py::dict());
        }
        py::dict s;
        s["decisions"] = action_stats_[team].decisions;
        s["applied"] = action_stats_[team].applied;
        s["wait"] = action_stats_[team].wait;
        py::dict rej;
        for (const auto& kv : action_stats_[team].rejections) {
            rej[py::str(kv.first)] = kv.second;
        }
        s["rejections"] = rej;
        action_stats.append(s);
    }
    info["reward_terms"] = reward_terms;
    info["action_debug"] = action_debug;
    info["action_stats"] = action_stats;
    info["freeze_on_action_timeout_s"] = static_cast<double>(ticks_per_step_) / static_cast<double>(std::max(1, tick_hz_));
    info["action_timeout_freeze_active"] = false;
    info["action_timeout_freeze_count"] = 0;
    info["waiting_for_step_sim_s"] = 0.0;
    std::array<int, 2> pending_by_team = {0, 0};
    for (const auto& p : pending_spawns_) {
        if (p.team >= 0 && p.team < 2) {
            pending_by_team[p.team] += 1;
        }
    }
    info["pending_spawns_total"] = static_cast<int>(pending_spawns_.size());
    py::list pending_team;
    pending_team.append(pending_by_team[0]);
    pending_team.append(pending_by_team[1]);
    info["pending_spawns_team"] = pending_team;
    info["spawns_started_this_step"] = py::list();
    info["spawns_activated_this_step"] = py::list();
    info["queue_hz"] = kQueueHz;
    info["queue_dt_s"] = kQueueDtS;
    info["spawn_min_delay_s"] = kSpawnMinDelayS;
    info["deploy_min_delay_s"] = kDeployMinDelayS;
    py::dict out;
    out["obs"] = obs;
    out["action_mask"] = action_mask;
    out["reward"] = reward;
    out["done"] = episode_done_;
    out["truncation"] = episode_truncated_;
    out["info"] = info;
    clear_step_reward_buffers();
    return out;
}
std::array<std::vector<float>, kAgentCount> ClashEnv::obs_vectors() const {
    std::array<std::vector<float>, kAgentCount> out;
    for (int team = 0; team < kAgentCount; ++team) {
        out[static_cast<size_t>(team)] = build_obs_vector(team);
    }
    return out;
}
std::array<std::vector<double>, kAgentCount> ClashEnv::action_masks() const {
    std::array<std::vector<double>, kAgentCount> out;
    for (int team = 0; team < kAgentCount; ++team) {
        out[static_cast<size_t>(team)] = action_mask_flat(team);
    }
    return out;
}
std::array<std::vector<std::vector<double>>, kAgentCount> ClashEnv::card_position_masks() const {
    std::array<std::vector<std::vector<double>>, kAgentCount> out;
    for (int team = 0; team < kAgentCount; ++team) {
        out[static_cast<size_t>(team)] = all_card_position_masks(team);
    }
    return out;
}
std::array<double, kAgentCount> ClashEnv::rewards_snapshot() const {
    return rewards_;
}
bool ClashEnv::done_flag() const {
    return episode_done_;
}
bool ClashEnv::truncation_flag() const {
    return episode_truncated_;
}
int ClashEnv::inferred_winner() const {
    if (!episode_done_ || episode_truncated_) {
        return -1;
    }
    const double t0_kill = reward_terms_[0].count("tower_kill") > 0 ? reward_terms_[0].at("tower_kill") : 0.0;
    const double t1_kill = reward_terms_[1].count("tower_kill") > 0 ? reward_terms_[1].at("tower_kill") : 0.0;
    if (t0_kill > t1_kill) {
        return 0;
    }
    if (t1_kill > t0_kill) {
        return 1;
    }
    return -1;
}
void ClashEnv::clear_step_reward_buffers() {
    rewards_[0] = 0.0;
    rewards_[1] = 0.0;
    reward_terms_[0].clear();
    reward_terms_[1].clear();
}
py::dict ClashEnv::debug_state() const {
    py::dict out;
    out["protocol_version"] = "knockoff_env_debug_v1";
    out["sim_time_s"] = sim_time_s_;
    out["tick_hz"] = tick_hz_;
    out["ticks_per_step"] = ticks_per_step_;
    out["done"] = episode_done_;
    out["truncation"] = episode_truncated_;
    py::list elixir;
    elixir.append(elixir_[0]);
    elixir.append(elixir_[1]);
    out["elixir"] = elixir;
    py::list infinite_elixir;
    infinite_elixir.append(infinite_elixir_[0]);
    infinite_elixir.append(infinite_elixir_[1]);
    out["infinite_elixir_teams"] = infinite_elixir;

    py::list entities;
    for (const auto& e : entities_) {
        py::dict d;
        d["id"] = e.id;
        d["kind"] = entity_kind_name(e.kind);
        d["team"] = e.team;
        d["card_id"] = e.card_id;
        d["alive"] = e.alive;
        d["x"] = e.x;
        d["y"] = e.y;
        d["hp"] = e.hp;
        d["max_hp"] = e.max_hp;
        d["radius"] = e.radius;
        d["lock_target_id"] = e.lock_target_id;
        d["stun_rem"] = e.stun_rem;
        d["deployment_lock_rem"] = e.deployment_lock_rem;
        entities.append(d);
    }
    out["entities"] = entities;

    py::list pending_spawns;
    for (const auto& p : pending_spawns_) {
        py::dict d;
        d["spawn_id"] = p.spawn_id;
        d["team"] = p.team;
        d["card_id"] = p.card_id;
        d["x"] = p.x;
        d["y"] = p.y;
        d["spawn_at_s"] = p.spawn_at_s;
        d["active_at_s"] = p.active_at_s;
        d["state"] = p.state;
        py::list entity_ids;
        for (int entity_id : p.entity_ids) {
            entity_ids.append(entity_id);
        }
        d["entity_ids"] = entity_ids;
        pending_spawns.append(d);
    }
    out["pending_spawns"] = pending_spawns;

    py::list fireballs;
    for (const auto& f : fireballs_) {
        py::dict d;
        d["team"] = f.team;
        d["x"] = f.x;
        d["y"] = f.y;
        d["detonate_at_s"] = f.detonate_at_s;
        fireballs.append(d);
    }
    out["fireballs"] = fireballs;

    py::list logs;
    for (const auto& s : logs_) {
        py::dict d;
        d["team"] = s.team;
        d["x"] = s.x;
        d["y"] = s.y;
        d["dir_y"] = s.dir_y;
        d["time_left_s"] = s.time_left_s;
        d["hit_count"] = s.hit_count;
        logs.append(d);
    }
    out["logs"] = logs;
    return out;
}
void ClashEnv::collect_tower_rewards() {
    const double current_team0 = sum_visible_tower_health(0);
    const double current_team1 = sum_visible_tower_health(1);
    const double damage_to_team0 = std::max(0.0, last_tower_health_[0] - current_team0);
    const double damage_to_team1 = std::max(0.0, last_tower_health_[1] - current_team1);
    if (damage_to_team1 > 0.0) {
        const double dealt_ratio = damage_to_team1 / std::max(1.0, initial_tower_health_[1]);
        const double dealt = clampd(dealt_ratio * kTowerHpDeltaRewardScale, 0.0, kTowerHpDeltaRewardClip);
        add_reward(0, dealt, "tower_damage_dealt");
        add_reward(1, -dealt, "tower_damage_taken");
    }
    if (damage_to_team0 > 0.0) {
        const double taken_ratio = damage_to_team0 / std::max(1.0, initial_tower_health_[0]);
        const double taken = clampd(taken_ratio * kTowerHpDeltaRewardScale, 0.0, kTowerHpDeltaRewardClip);
        add_reward(0, -taken, "tower_damage_taken");
        add_reward(1, taken, "tower_damage_dealt");
    }
    last_tower_health_[0] = current_team0;
    last_tower_health_[1] = current_team1;
}
} // namespace knockoff_cr
