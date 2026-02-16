#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace py = pybind11;
namespace knockoff_cr {
inline constexpr int kAgentCount = 2;
inline constexpr int kGridW = 18;
inline constexpr int kGridH = 32;
inline constexpr int kHalfH = kGridH / 2;
inline constexpr int kRiverTopRow = 15;
inline constexpr int kRiverBottomRow = 16;
inline constexpr double kBridgeSideMarginTiles = 2.5;
inline constexpr double kBridgeWidthTiles = 2.0;
inline constexpr int kUsableCardCount = 8;
inline constexpr int kHandSize = 4;
inline constexpr int kMaxEntitiesPerTeamObs = 8;
inline constexpr double kTowerHpDeltaRewardScale = 0.6;
inline constexpr double kTowerHpDeltaRewardClip = 0.05;
inline constexpr double kTerminalReward = 3.0;
inline constexpr int kPlacementRegionCols = 3;
inline constexpr int kPlacementRegionRows = 3;
inline constexpr double kSpawnMinDelayS = 1.0;
inline constexpr double kDeployMinDelayS = 0.5;
inline constexpr double kQueueHz = 10.0;
inline constexpr double kQueueDtS = 1.0 / kQueueHz;
inline constexpr double kTowerAttackRangeTiles = 8.5;
inline constexpr double kTowerAttackCooldownS = 0.8;
inline constexpr int kTowerDamage = 160;
inline constexpr double kSenseRadiusTiles = 8.0;
inline constexpr int kCardIceSpirit = 0;
inline constexpr int kCardMusketeer = 1;
inline constexpr int kCardCannon = 2;
inline constexpr int kCardHog = 3;
inline constexpr int kCardSkeletons = 4;
inline constexpr int kCardFireball = 5;
inline constexpr int kCardIceGolem = 6;
inline constexpr int kCardLog = 7;
enum CardType {
    CARD_TROOP = 0,
    CARD_BUILDING = 1,
    CARD_SPELL = 2,
};
enum TargetType {
    TARGET_GROUND = 0,
    TARGET_AIR = 1,
    TARGET_BOTH = 2,
    TARGET_BUILDINGS = 3,
};
enum EntityKind {
    ENTITY_TROOP = 0,
    ENTITY_BUILDING = 1,
    ENTITY_TOWER = 2,
};
struct CardDef {
    std::string name;
    int type = CARD_TROOP;
    int target_type = TARGET_BOTH;
    int cost = 1;
    int amount = 1;
    double speed = 0.0;
    int health = 1;
    int damage = 0;
    int tower_damage = 0;
    double attack_swing = 0.0;
    double attack_end = 0.8;
    double attack_range = 1.0;
    double decay_time = 30.0;
};
struct Entity {
    int id = -1;
    EntityKind kind = ENTITY_TROOP;
    int team = 0;
    int card_id = -1;
    bool king_tower = false;
    bool tower_active = false;
    double x = 0.0;
    double y = 0.0;
    double hp = 1.0;
    double max_hp = 1.0;
    double speed = 0.0;
    int damage = 0;
    int tower_damage = 0;
    double attack_range = 0.0;
    double attack_swing = 0.0;
    double attack_end = 0.0;
    int target_type = TARGET_BOTH;
    double decay_time = 0.0;
    double radius = 0.35;
    bool alive = true;
    double stun_rem = 0.0;
    double deployment_lock_rem = 0.0;
    int lock_target_id = -1;
    double attack_windup_rem = 0.0;
    double attack_recover_rem = 0.0;
    double tower_cooldown_rem = 0.0;
};
struct PendingSpawn {
    int spawn_id = -1;
    int team = 0;
    int card_id = -1;
    double x = 0.0;
    double y = 0.0;
    double spawn_at_s = 0.0;
    double active_at_s = -1.0;
    std::string state = "queued";
    std::vector<int> entity_ids;
};
struct FireballEvent {
    int team = 0;
    int card_id = kCardFireball;
    double x = 0.0;
    double y = 0.0;
    double detonate_at_s = 0.0;
};
struct LogSpell {
    int team = 0;
    int card_id = kCardLog;
    double x = 0.0;
    double y = 0.0;
    double dir_y = 0.0;
    double time_left_s = 2.0;
    int hit_count = 0;
    std::unordered_set<int> hit_entity_ids;
};
struct TeamStats {
    int decisions = 0;
    int applied = 0;
    int wait = 0;
    std::unordered_map<std::string, int> rejections;
};
struct ActionDebug {
    bool valid = false;
    int team = 0;
    int wait = 1;
    int card_selection = -1;
    int position_region = 0;
    int position_cell = 0;
    int grid_x = 0;
    int grid_y = 0;
    double world_x = 0.0;
    double world_y = 0.0;
    bool applied = false;
    std::string reason = "unknown";
    bool queued = false;
    int spawn_id = -1;
    double queue_hz = kQueueHz;
    double spawn_min_delay_s = kSpawnMinDelayS;
    double deploy_min_delay_s = kDeployMinDelayS;
};
struct PlayResult {
    bool applied = false;
    std::string reason = "unknown";
    int spawn_id = -1;
};
struct ExternalAction {
    int wait = 1;
    int card_selection = -1;
    int position_region = 0;
    int position_cell = 0;
};
struct StepCoreOutcome {
    std::array<bool, 2> action_applied = {false, false};
    int ticks_elapsed = 0;
};
std::vector<CardDef> build_cards();
double clampd(double v, double lo, double hi);
bool is_finite(double v);
double dist2(double ax, double ay, double bx, double by);
ExternalAction parse_external_action_dict(const py::dict& action);
class ClashEnv {
public:
    ClashEnv(int tick_hz = 10, double max_sim_seconds = 120.0, int seed = 1);
    py::dict spec();
    py::dict reset(int seed_value = -1, py::dict options = py::dict());
    py::dict step(py::list actions);
    StepCoreOutcome step_core(const std::array<ExternalAction, kAgentCount>& actions);
    py::dict finalize_step_transition(const StepCoreOutcome& outcome);
    std::array<std::vector<float>, kAgentCount> obs_vectors() const;
    std::array<std::vector<double>, kAgentCount> action_masks() const;
    std::array<std::vector<std::vector<double>>, kAgentCount> card_position_masks() const;
    std::array<double, kAgentCount> rewards_snapshot() const;
    bool done_flag() const;
    bool truncation_flag() const;
    int inferred_winner() const;
    void clear_step_reward_buffers();
    py::dict debug_state() const;
private:
    int tick_hz_ = 10;
    double dt_ = 0.1;
    double max_sim_seconds_ = 120.0;
    int ticks_per_step_ = 10;
    std::vector<CardDef> cards_;
    std::mt19937 rng_;
    std::vector<Entity> entities_;
    std::vector<PendingSpawn> pending_spawns_;
    std::vector<FireballEvent> fireballs_;
    std::vector<LogSpell> logs_;
    std::array<std::vector<int>, 2> decks_;
    std::array<double, 2> elixir_ = {8.0, 8.0};
    std::array<bool, 2> infinite_elixir_ = {false, false};
    double base_seconds_per_elixir_ = 2.8;
    double max_elixir_ = 10.0;
    bool episode_done_ = false;
    bool episode_truncated_ = false;
    bool reset_requested_ = false;
    std::array<double, 2> rewards_ = {0.0, 0.0};
    std::array<std::unordered_map<std::string, double>, 2> reward_terms_;
    std::array<ActionDebug, 2> last_action_debug_;
    std::array<TeamStats, 2> action_stats_;
    std::array<std::string, 2> team_controllers_ = {"external", "external"};
    std::array<double, 2> initial_tower_health_ = {1.0, 1.0};
    std::array<double, 2> last_tower_health_ = {1.0, 1.0};
    double time_since_tower_damaged_ = 0.0;
    double time_since_tower_reward_ = 0.0;
    double sim_time_s_ = 0.0;
    double queue_accum_s_ = 0.0;
    double queue_time_s_ = 0.0;
    int next_entity_id_ = 1;
    int next_spawn_id_ = 1;
    int region_width() const;
    int region_height() const;
    int placement_region_count() const;
    int placement_cells_per_region() const;
    void add_reward(int team, double delta, const std::string& term);
    std::pair<int, int> action_to_internal_grid(int team, int action_x, int action_y) const;
    std::pair<double, double> internal_to_world(int gx, int gy) const;
    std::pair<int, int> world_to_internal_grid(double x, double y) const;
    bool is_bridge_internal_x(int gx) const;
    bool is_bridge_world_x(double x) const;
    double bridge_left_center_x() const;
    double bridge_right_center_x() const;
    bool is_water_internal(int gy) const;
    bool is_alive(const Entity& e) const;
    Entity* find_entity(int id);
    const Entity* find_entity_const(int id) const;
    void reset_state();
    void add_tower(int team, bool king, bool active, int gx, int gy, double hp);
    std::vector<int> hand_for_team(int team) const;
    bool team_has_infinite_elixir(int team) const;
    bool card_in_hand(int team, int card_id) const;
    void cycle_deck_after_play(int team, int card_id);
    bool legal_placement_for_card(int team, const CardDef& card, int action_x, int action_y) const;
    int fill_legal_positions(int team, const CardDef& card, std::vector<double>* mask_out) const;
    std::vector<double> position_mask_for_card(int team, int card_id) const;
    std::vector<double> action_mask_flat(int team) const;
    int count_alive_towers(int team) const;
    double sum_visible_tower_health(int team) const;
    int count_playable_cards(int team) const;
    std::vector<float> build_obs_vector(int for_team) const;
    std::vector<std::vector<double>> all_card_position_masks(int team) const;
    py::dict build_transition(py::dict info, py::tuple action_applied, int ticks_simulated, int ticks_requested);
    void collect_tower_rewards();
    bool apply_action(int team, const py::dict& action);
    bool apply_action(int team, const ExternalAction& action);
    PlayResult play_card_with_reason(int team, int card_id, int action_x, int action_y);
    void simulate_tick();
    void process_queue_tick();
    std::vector<int> spawn_card_entities(int team, int card_id, double x, double y);
    bool can_target(const Entity& attacker, const Entity& target) const;
    Entity* nearest_target_within(const Entity& attacker, double max_range);
    Entity* nearest_princess_tower(const Entity& seeker, int enemy_team);
    bool target_in_attack_range(const Entity& attacker, const Entity& target) const;
    std::pair<double, double> movement_target(const Entity& e);
    std::pair<double, double> bridge_waypoint_if_needed(const Entity& e, double tx, double ty);
    bool cell_walkable_for(const Entity& e, int gx, int gy) const;
    bool line_walkable_for(const Entity& e, double x0, double y0, double x1, double y1) const;
    std::pair<double, double> next_path_waypoint(const Entity& e, double tx, double ty) const;
    void apply_damage(Entity& target, int dmg, int source_team, int source_card, bool from_spell);
    void resolve_spell_reward(int caster_team, int hit_count);
    void process_fireballs();
    void process_logs();
    void update_entities();
    void update_tower(Entity& e);
    void begin_attack_if_possible(Entity& e, Entity& target);
    void resolve_attack_hit(Entity& e);
    void update_stationary_attacker(Entity& e);
    void update_troop(Entity& e);
    void process_push_collision();
    void cleanup_dead_entities();
};

class ClashEnvBatch {
public:
    ClashEnvBatch(
        int num_envs,
        int tick_hz = 10,
        double max_sim_seconds = 120.0,
        int seed = 1,
        int num_threads = 0);

    py::dict spec();
    py::list reset_many(py::object seeds = py::none(), py::object options_per_env = py::none());
    py::list step_many(py::list actions_per_env);
    py::list step_many_discrete(py::array_t<int, py::array::c_style | py::array::forcecast> actions_per_env);
    py::dict step_many_packed(py::array_t<int, py::array::c_style | py::array::forcecast> actions_per_env);
    py::list debug_state_many() const;

private:
    int num_threads_ = 1;
    std::vector<ClashEnv> envs_;
};
} // namespace knockoff_cr
