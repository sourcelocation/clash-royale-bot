#include "knockoff_cr/backend.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace knockoff_cr {
namespace {
int clamp_thread_count(int requested, int env_count) {
    if (env_count <= 0) {
        return 1;
    }
    if (requested > 0) {
        return std::max(1, std::min(requested, env_count));
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    const int auto_threads = hw > 0 ? static_cast<int>(hw) : 1;
    return std::max(1, std::min(auto_threads, env_count));
}

std::vector<std::array<ExternalAction, kAgentCount>> parse_discrete_action_tensor(
    py::array_t<int, py::array::c_style | py::array::forcecast> actions_per_env,
    int env_count,
    const char* fn_name) {
    py::buffer_info info = actions_per_env.request();
    if (info.ndim != 3) {
        throw std::runtime_error(std::string(fn_name) + " expects rank-3 array [env,agent,branch]");
    }
    if (info.shape[0] != env_count || info.shape[1] != kAgentCount || info.shape[2] != 4) {
        throw std::runtime_error(std::string(fn_name) + " shape mismatch; expected [num_envs,2,4]");
    }

    const auto* raw = static_cast<const int*>(info.ptr);
    std::vector<std::array<ExternalAction, kAgentCount>> parsed_actions(static_cast<size_t>(env_count));
    for (int env_idx = 0; env_idx < env_count; ++env_idx) {
        for (int team = 0; team < kAgentCount; ++team) {
            const size_t base = static_cast<size_t>(env_idx * kAgentCount * 4 + team * 4);
            ExternalAction parsed;
            parsed.wait = raw[base + 0];
            parsed.card_selection = raw[base + 1];
            parsed.position_region = raw[base + 2];
            parsed.position_cell = raw[base + 3];
            parsed_actions[static_cast<size_t>(env_idx)][static_cast<size_t>(team)] = parsed;
        }
    }
    return parsed_actions;
}

std::vector<StepCoreOutcome> run_step_core_parallel(
    std::vector<ClashEnv>& envs,
    const std::vector<std::array<ExternalAction, kAgentCount>>& parsed_actions,
    int num_threads) {
    const int env_count = static_cast<int>(envs.size());
    std::vector<StepCoreOutcome> outcomes(static_cast<size_t>(env_count));
    std::mutex err_mu;
    std::exception_ptr first_err = nullptr;
    std::atomic<int> next_idx{0};
    const int threads = std::max(1, std::min(num_threads, env_count));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads));

    auto worker = [&]() {
        while (true) {
            const int i = next_idx.fetch_add(1);
            if (i >= env_count) {
                return;
            }
            try {
                outcomes[static_cast<size_t>(i)] = envs[static_cast<size_t>(i)].step_core(parsed_actions[static_cast<size_t>(i)]);
            } catch (...) {
                std::lock_guard<std::mutex> lock(err_mu);
                if (!first_err) {
                    first_err = std::current_exception();
                }
                return;
            }
        }
    };

    {
        py::gil_scoped_release release;
        for (int t = 0; t < threads; ++t) {
            workers.emplace_back(worker);
        }
        for (auto& th : workers) {
            th.join();
        }
    }
    if (first_err) {
        std::rethrow_exception(first_err);
    }
    return outcomes;
}

py::list finalize_step_list(std::vector<ClashEnv>& envs, const std::vector<StepCoreOutcome>& outcomes) {
    py::list result;
    for (size_t i = 0; i < envs.size(); ++i) {
        result.append(envs[i].finalize_step_transition(outcomes[i]));
    }
    return result;
}
} // namespace

ClashEnvBatch::ClashEnvBatch(int num_envs, int tick_hz, double max_sim_seconds, int seed, int num_threads)
    : num_threads_(clamp_thread_count(num_threads, std::max(1, num_envs))) {
    const int count = std::max(1, num_envs);
    envs_.reserve(static_cast<size_t>(count));
    for (int env_id = 0; env_id < count; ++env_id) {
        envs_.emplace_back(tick_hz, max_sim_seconds, seed + env_id);
    }
}

py::dict ClashEnvBatch::spec() {
    return envs_.front().spec();
}

py::list ClashEnvBatch::reset_many(py::object seeds, py::object options_per_env) {
    const int env_count = static_cast<int>(envs_.size());
    std::vector<int> parsed_seeds(static_cast<size_t>(env_count), -1);
    std::vector<py::dict> parsed_options;
    parsed_options.reserve(static_cast<size_t>(env_count));
    std::vector<bool> should_reset(static_cast<size_t>(env_count), true);

    {
        py::gil_scoped_acquire gil;
        for (int i = 0; i < env_count; ++i) {
            parsed_options.emplace_back(py::dict());
        }
        if (!seeds.is_none()) {
            py::list seed_list = py::cast<py::list>(seeds);
            if (py::len(seed_list) != env_count) {
                throw std::runtime_error("reset_many seeds length mismatch");
            }
            for (int i = 0; i < env_count; ++i) {
                py::object seed_obj = seed_list[i];
                if (!seed_obj.is_none()) {
                    parsed_seeds[static_cast<size_t>(i)] = py::cast<int>(seed_obj);
                }
            }
        }

        if (!options_per_env.is_none()) {
            py::list options_list = py::cast<py::list>(options_per_env);
            if (py::len(options_list) != env_count) {
                throw std::runtime_error("reset_many options_per_env length mismatch");
            }
            for (int i = 0; i < env_count; ++i) {
                py::object opt_obj = options_list[i];
                if (opt_obj.is_none()) {
                    should_reset[static_cast<size_t>(i)] = false;
                    continue;
                }
                parsed_options[static_cast<size_t>(i)] = py::cast<py::dict>(opt_obj);
            }
        }
    }

    py::list result;
    for (int i = 0; i < env_count; ++i) {
        if (should_reset[static_cast<size_t>(i)]) {
            result.append(envs_[static_cast<size_t>(i)].reset(
                parsed_seeds[static_cast<size_t>(i)],
                parsed_options[static_cast<size_t>(i)]));
        } else {
            result.append(py::none());
        }
    }
    return result;
}

py::list ClashEnvBatch::step_many(py::list actions_per_env) {
    const int env_count = static_cast<int>(envs_.size());
    if (py::len(actions_per_env) != env_count) {
        throw std::runtime_error("step_many actions_per_env length mismatch");
    }
    std::vector<std::array<ExternalAction, kAgentCount>> parsed_actions(static_cast<size_t>(env_count));
    for (int env_idx = 0; env_idx < env_count; ++env_idx) {
        py::list per_env = py::cast<py::list>(actions_per_env[env_idx]);
        if (py::len(per_env) != kAgentCount) {
            throw std::runtime_error("step_many per-env action agent count mismatch");
        }
        for (int team = 0; team < kAgentCount; ++team) {
            const py::dict action = py::cast<py::dict>(per_env[team]);
            parsed_actions[static_cast<size_t>(env_idx)][static_cast<size_t>(team)] = parse_external_action_dict(action);
        }
    }
    const std::vector<StepCoreOutcome> outcomes = run_step_core_parallel(envs_, parsed_actions, num_threads_);
    return finalize_step_list(envs_, outcomes);
}

py::list ClashEnvBatch::step_many_discrete(py::array_t<int, py::array::c_style | py::array::forcecast> actions_per_env) {
    const int env_count = static_cast<int>(envs_.size());
    const auto parsed_actions = parse_discrete_action_tensor(actions_per_env, env_count, "step_many_discrete");
    const std::vector<StepCoreOutcome> outcomes = run_step_core_parallel(envs_, parsed_actions, num_threads_);
    return finalize_step_list(envs_, outcomes);
}

py::dict ClashEnvBatch::step_many_packed(py::array_t<int, py::array::c_style | py::array::forcecast> actions_per_env) {
    const int env_count = static_cast<int>(envs_.size());
    const auto parsed_actions = parse_discrete_action_tensor(actions_per_env, env_count, "step_many_packed");
    run_step_core_parallel(envs_, parsed_actions, num_threads_);

    const int obs_dim = static_cast<int>(envs_.front().obs_vectors()[0].size());
    const int mask_dim = static_cast<int>(envs_.front().action_masks()[0].size());
    const int cards = kUsableCardCount;
    const int positions = kGridW * kGridH;
    py::array_t<float> obs({env_count, kAgentCount, obs_dim});
    py::array_t<float> action_mask({env_count, kAgentCount, mask_dim});
    py::array_t<float> card_position_masks({env_count, kAgentCount, cards, positions});
    py::array_t<float> reward({env_count, kAgentCount});
    py::array_t<uint8_t> done({env_count});
    py::array_t<uint8_t> truncation({env_count});
    py::array_t<int8_t> winner({env_count});

    auto obs_view = obs.mutable_unchecked<3>();
    auto mask_view = action_mask.mutable_unchecked<3>();
    auto card_view = card_position_masks.mutable_unchecked<4>();
    auto reward_view = reward.mutable_unchecked<2>();
    auto done_view = done.mutable_unchecked<1>();
    auto trunc_view = truncation.mutable_unchecked<1>();
    auto winner_view = winner.mutable_unchecked<1>();

    for (int env_idx = 0; env_idx < env_count; ++env_idx) {
        auto& env = envs_[static_cast<size_t>(env_idx)];
        const auto obs_by_team = env.obs_vectors();
        const auto mask_by_team = env.action_masks();
        const auto cards_by_team = env.card_position_masks();
        const auto rewards = env.rewards_snapshot();
        for (int team = 0; team < kAgentCount; ++team) {
            for (int j = 0; j < obs_dim; ++j) {
                obs_view(env_idx, team, j) = obs_by_team[static_cast<size_t>(team)][static_cast<size_t>(j)];
            }
            for (int j = 0; j < mask_dim; ++j) {
                mask_view(env_idx, team, j) = static_cast<float>(mask_by_team[static_cast<size_t>(team)][static_cast<size_t>(j)]);
            }
            for (int c = 0; c < cards; ++c) {
                for (int p = 0; p < positions; ++p) {
                    card_view(env_idx, team, c, p) = static_cast<float>(
                        cards_by_team[static_cast<size_t>(team)][static_cast<size_t>(c)][static_cast<size_t>(p)]
                    );
                }
            }
            reward_view(env_idx, team) = static_cast<float>(rewards[static_cast<size_t>(team)]);
        }
        done_view(env_idx) = env.done_flag() ? 1 : 0;
        trunc_view(env_idx) = env.truncation_flag() ? 1 : 0;
        winner_view(env_idx) = static_cast<int8_t>(env.inferred_winner());
        env.clear_step_reward_buffers();
    }

    py::dict out;
    out["obs"] = obs;
    out["action_mask"] = action_mask;
    out["card_position_masks"] = card_position_masks;
    out["reward"] = reward;
    out["done"] = done;
    out["truncation"] = truncation;
    out["winner"] = winner;
    return out;
}

py::list ClashEnvBatch::debug_state_many() const {
    py::list out;
    for (const auto& env : envs_) {
        out.append(env.debug_state());
    }
    return out;
}

} // namespace knockoff_cr
