#include "knockoff_cr/backend.hpp"

#include <algorithm>
#include <array>

namespace knockoff_cr {
namespace {
double troop_radius_for_card(int card_id) {
    switch (card_id) {
        case kCardIceSpirit:
            return 0.5;
        case kCardSkeletons:
            return 0.35;
        case kCardMusketeer:
            return 0.5;
        case kCardHog:
            return 0.75;
        case kCardIceGolem:
            return 0.75;
        default:
            return 0.5;
    }
}

double building_radius_for_card(int card_id) {
    switch (card_id) {
        case kCardCannon:
            return 1.0;
        default:
            return 0.90;
    }
}
} // namespace

void ClashEnv::simulate_tick() {
    if (episode_done_ || episode_truncated_) {
        return;
    }

    sim_time_s_ += dt_;
    queue_accum_s_ += dt_;
    time_since_tower_damaged_ += dt_;
    time_since_tower_reward_ += dt_;

    elixir_[0] = clampd(elixir_[0] + dt_ / base_seconds_per_elixir_, 0.0, max_elixir_);
    elixir_[1] = clampd(elixir_[1] + dt_ / base_seconds_per_elixir_, 0.0, max_elixir_);

    while (queue_accum_s_ >= kQueueDtS) {
        queue_accum_s_ -= kQueueDtS;
        queue_time_s_ += kQueueDtS;
        process_queue_tick();
    }

    if (time_since_tower_reward_ >= 0.2) {
        time_since_tower_reward_ = 0.0;
        collect_tower_rewards();
    }

    process_fireballs();
    process_logs();
    update_entities();
    process_push_collision();

    cleanup_dead_entities();

    if (!episode_done_ && sim_time_s_ >= max_sim_seconds_) {
        episode_truncated_ = true;
        reset_requested_ = true;
    }
}

void ClashEnv::process_queue_tick() {
    std::vector<int> remove_indices;
    remove_indices.reserve(pending_spawns_.size());

    for (int i = 0; i < static_cast<int>(pending_spawns_.size()); ++i) {
        auto& p = pending_spawns_[i];
        if (p.state == "queued") {
            if (queue_time_s_ < p.spawn_at_s) {
                continue;
            }

            if (p.card_id < 0 || p.card_id >= kUsableCardCount) {
                remove_indices.push_back(i);
                continue;
            }

            const CardDef& card = cards_[p.card_id];

            if (card.type == CARD_TROOP || card.type == CARD_BUILDING) {
                auto created = spawn_card_entities(p.team, p.card_id, p.x, p.y);
                p.entity_ids = created;
                p.state = "deploying";
                p.active_at_s = queue_time_s_ + kDeployMinDelayS;
            } else {
                if (p.card_id == kCardFireball) {
                    FireballEvent f;
                    f.team = p.team;
                    f.card_id = p.card_id;
                    f.x = p.x;
                    f.y = p.y;
                    f.detonate_at_s = queue_time_s_ + 1.0;
                    fireballs_.push_back(f);
                } else if (p.card_id == kCardLog) {
                    LogSpell s;
                    s.team = p.team;
                    s.card_id = p.card_id;
                    s.x = p.x;
                    s.y = p.y;
                    s.time_left_s = 2.0;
                    s.dir_y = (p.team == 0) ? -1.0 : 1.0;
                    logs_.push_back(s);
                }
                remove_indices.push_back(i);
            }
        } else if (p.state == "deploying") {
            if (queue_time_s_ < p.active_at_s) {
                continue;
            }
            for (int entity_id : p.entity_ids) {
                if (Entity* e = find_entity(entity_id)) {
                    e->deployment_lock_rem = 0.0;
                }
            }
            remove_indices.push_back(i);
        } else {
            remove_indices.push_back(i);
        }
    }

    std::sort(remove_indices.begin(), remove_indices.end());
    remove_indices.erase(std::unique(remove_indices.begin(), remove_indices.end()), remove_indices.end());
    for (int idx_i = static_cast<int>(remove_indices.size()) - 1; idx_i >= 0; --idx_i) {
        const int idx = remove_indices[idx_i];
        if (idx >= 0 && idx < static_cast<int>(pending_spawns_.size())) {
            pending_spawns_.erase(pending_spawns_.begin() + idx);
        }
    }
}

std::vector<int> ClashEnv::spawn_card_entities(int team, int card_id, double x, double y) {
    std::vector<int> out;
    if (card_id < 0 || card_id >= kUsableCardCount) {
        return out;
    }

    const CardDef& card = cards_[card_id];
    const int amount = std::max(1, card.amount);

    for (int i = 0; i < amount; ++i) {
        Entity e;
        e.id = next_entity_id_++;
        e.kind = (card.type == CARD_BUILDING) ? ENTITY_BUILDING : ENTITY_TROOP;
        e.team = team;
        e.card_id = card_id;
        e.hp = static_cast<double>(card.health);
        e.max_hp = static_cast<double>(card.health);
        e.damage = card.damage;
        e.tower_damage = card.tower_damage;
        e.speed = card.speed;
        e.attack_range = card.attack_range;
        e.attack_swing = card.attack_swing;
        e.attack_end = card.attack_end;
        e.target_type = card.target_type;
        e.decay_time = (e.kind == ENTITY_BUILDING) ? card.decay_time : 0.0;
        e.radius = (e.kind == ENTITY_BUILDING) ? building_radius_for_card(card_id) : troop_radius_for_card(card_id);
        e.deployment_lock_rem = kDeployMinDelayS;

        double sx = x;
        double sy = y;
        if (card_id == kCardSkeletons) {
            static const std::array<std::pair<double, double>, 3> kOffsets = {{{-0.25, 0.0}, {0.25, 0.0}, {0.0, 0.25}}};
            const auto [ox, oy] = kOffsets[std::min(i, 2)];
            sx += ox;
            sy += oy;
        }

        e.x = clampd(sx, -(kGridW / 2.0), (kGridW / 2.0));
        e.y = clampd(sy, -(kGridH / 2.0), (kGridH / 2.0));

        entities_.push_back(e);
        out.push_back(e.id);
    }

    return out;
}

} // namespace knockoff_cr
