extends Node
class_name ArenaEnvAdapter

const PROTOCOL_VERSION := "knockoff_env_v1"
const OBS_VERSION := "v1"
const ENV_SCHEMA_VERSION := "knockoff_cr_env_v2"
const AGENT_COUNT := 2

var _game: Node = null
var _ticks_per_step: int = 1

var _observation_encoder: ObservationEncoder = ObservationEncoder.new()
var _action_masker: ActionMasker = ActionMasker.new()
var _reward_engine: RewardEngine = RewardEngine.new()
var _termination_engine: TerminationEngine = TerminationEngine.new()

var _step_in_flight: bool = false
var _step_ready: bool = false
var _step_result: Dictionary = {}
var _pending_action_applied: Array = [false, false]
var _pending_step_ticks_total: int = 1
var _pending_step_ticks_elapsed: int = 0
var _waiting_for_step_sim_s: float = 0.0
var _timeout_freeze_active: bool = false
var _timeout_prev_time_scale: float = 1.0
var _timeout_freeze_count: int = 0
var _timeout_freeze_started_wall_s: float = 0.0


func _ready() -> void:
	add_child(_observation_encoder)
	add_child(_action_masker)
	add_child(_reward_engine)
	add_child(_termination_engine)


func _physics_process(_delta: float) -> void:
	if not _step_in_flight:
		_waiting_for_step_sim_s += max(0.0, _delta)
		var timeout_sim_s = float(_ticks_per_step) / float(max(1, Engine.physics_ticks_per_second))
		if not _timeout_freeze_active and _waiting_for_step_sim_s >= timeout_sim_s:
			_activate_timeout_freeze()
		return

	_waiting_for_step_sim_s = 0.0
	_pending_step_ticks_elapsed += 1
	var done_info = _termination_engine.consume_done_info()
	var terminal = bool(done_info.get("done", false)) or bool(done_info.get("truncation", false))
	var reached_tick_budget = _pending_step_ticks_elapsed >= _pending_step_ticks_total
	if not reached_tick_budget and not terminal:
		return

	var info = {
		"action_applied": _pending_action_applied.duplicate(),
		"ticks_simulated": _pending_step_ticks_elapsed,
		"ticks_requested": _pending_step_ticks_total,
	}
	if terminal and _pending_step_ticks_elapsed < _pending_step_ticks_total:
		info["terminated_early"] = true
	_step_result = _build_transition(info)
	_step_ready = true
	_step_in_flight = false


func bind_game(game: Node) -> void:
	_game = game
	if _game and _game.has_method("env_set_external_reset_control"):
		_game.env_set_external_reset_control(true)
	if _game and _game.has_method("env_set_training_mode"):
		_game.env_set_training_mode(true)
	if _game and _game.has_method("env_set_matchup"):
		_game.env_set_matchup("external", "external")
	_observation_encoder.bind_game(game)
	_action_masker.bind_game(game)
	_reward_engine.bind_game(game)
	_termination_engine.bind_game(game)


func get_spec() -> Dictionary:
	var action_space = _game.env_get_action_space() if _game else {}
	var sample_obs = _observation_encoder.encode(0)
	var sample_action_mask = _action_masker.encode_flat(0)
	var placement_schema = _game.env_get_placement_schema() if _game and _game.has_method("env_get_placement_schema") else {}
	var sample_card_masks = sample_obs.get("position_masks_for_all_cards", [])
	var cards_count = len(sample_card_masks)
	var per_card_positions = 0
	if cards_count > 0 and sample_card_masks[0] is Array:
		per_card_positions = len(sample_card_masks[0])

	return {
		"protocol_version": PROTOCOL_VERSION,
		"obs_version": OBS_VERSION,
		"schema_version": ENV_SCHEMA_VERSION,
		"n_agents": AGENT_COUNT,
		"action_space": action_space,
		"action_order": _game.ACTION_ORDER if _game else [],
		"obs_schema": {
			"vector_size": len(sample_obs.get("vector", [])),
			"position_masks_cards": cards_count,
			"position_masks_per_card": per_card_positions,
			"spatial_shape": [0, 0, 0],
		},
		"action_mask_size": len(sample_action_mask),
		"ticks_per_step": _ticks_per_step,
		"placement_schema": placement_schema,
	}


func reset(seed_value: int = -1, options: Dictionary = {}) -> Dictionary:
	if seed_value >= 0:
		seed(seed_value)

	var parsed_options = _parse_reset_options(options)
	_ticks_per_step = max(1, int(parsed_options.get("ticks_per_step", _ticks_per_step)))
	_waiting_for_step_sim_s = 0.0
	_deactivate_timeout_freeze()
	_clear_pending_step_state()
	if _game:
		if _game.has_method("env_set_training_mode"):
			_game.env_set_training_mode(bool(parsed_options.get("training_mode", true)))
		if _game.has_method("env_set_matchup"):
			var controllers: Array = parsed_options.get("team_controllers", ["external", "external"])
			_game.env_set_matchup(str(controllers[0]), str(controllers[1]))
		_game.env_reset(seed_value)

	var info = {"reset": true, "options": parsed_options}
	return _build_transition(info)

func _parse_reset_options(options: Dictionary) -> Dictionary:
	var parsed = options.duplicate(true)
	var default_controllers = ["external", "external"]
	var controllers: Array = default_controllers
	if parsed.has("team_controllers"):
		var requested = parsed.get("team_controllers")
		if requested is Array and requested.size() >= 2:
			controllers = [str(requested[0]).to_lower(), str(requested[1]).to_lower()]
	parsed["team_controllers"] = controllers
	if not parsed.has("training_mode"):
		parsed["training_mode"] = true
	if not parsed.has("ticks_per_step"):
		parsed["ticks_per_step"] = _ticks_per_step
	parsed["ticks_per_step"] = max(1, int(parsed.get("ticks_per_step", _ticks_per_step)))
	return parsed


func begin_step(actions: Array) -> Dictionary:
	if _step_in_flight:
		return {
			"accepted": false,
			"data": _build_transition(
				{
					"action_applied": [false, false],
					"error": "step_in_flight",
				}
			),
		}

	if actions.size() != AGENT_COUNT:
		return {
			"accepted": false,
			"data": _build_transition(
				{
					"action_applied": [false, false],
					"error": "invalid_action_count",
					"expected_actions": AGENT_COUNT,
					"received_actions": actions.size(),
				}
			),
		}

	var action_applied: Array = [false, false]
	if _timeout_freeze_active:
		_deactivate_timeout_freeze()
	_waiting_for_step_sim_s = 0.0
	if _game:
		for team in range(AGENT_COUNT):
			if team >= actions.size():
				continue
			var team_action = actions[team]
			if team_action is Dictionary:
				action_applied[team] = _game.env_step(team, team_action)

	_pending_action_applied = action_applied
	_pending_step_ticks_total = max(1, _ticks_per_step)
	_pending_step_ticks_elapsed = 0
	_step_in_flight = true
	_step_ready = false
	_step_result = {}
	return {"accepted": true}


func has_step_result() -> bool:
	return _step_ready


func take_step_result() -> Dictionary:
	if not _step_ready:
		return _build_transition(
			{
				"action_applied": [false, false],
				"error": "step_result_not_ready",
			}
		)
	var result = _step_result
	_step_result = {}
	_step_ready = false
	return result


func _clear_pending_step_state() -> void:
	_step_in_flight = false
	_step_ready = false
	_step_result = {}
	_pending_action_applied = [false, false]
	_pending_step_ticks_total = max(1, _ticks_per_step)
	_pending_step_ticks_elapsed = 0
	_waiting_for_step_sim_s = 0.0


func _activate_timeout_freeze() -> void:
	if _timeout_freeze_active:
		return
	_timeout_prev_time_scale = Engine.time_scale
	Engine.time_scale = 0.0
	_timeout_freeze_active = true
	_timeout_freeze_count += 1
	_timeout_freeze_started_wall_s = float(Time.get_ticks_usec()) / 1000000.0


func _deactivate_timeout_freeze() -> void:
	if not _timeout_freeze_active:
		return
	var now_wall_s = float(Time.get_ticks_usec()) / 1000000.0
	var frozen_wall_s = max(0.0, now_wall_s - _timeout_freeze_started_wall_s)
	print("Was frozen for %.3fs" % frozen_wall_s)
	Engine.time_scale = max(0.01, _timeout_prev_time_scale)
	_timeout_freeze_active = false
	_timeout_freeze_started_wall_s = 0.0


func _build_transition(info: Dictionary) -> Dictionary:
	var obs: Array = []
	var action_mask_flat: Array = []
	var reward: Array = []
	var reward_terms: Array = []
	var action_debug: Array = []
	var action_stats: Array = []
	for team in range(AGENT_COUNT):
		obs.append(_observation_encoder.encode(team))
		action_mask_flat.append(_action_masker.encode_flat(team))
		var reward_payload = _reward_engine.consume_reward_payload(team)
		reward.append(float(reward_payload.get("reward", 0.0)))
		reward_terms.append(reward_payload.get("terms", {}))
		if _game and _game.has_method("env_consume_action_debug"):
			action_debug.append(_game.env_consume_action_debug(team))
		else:
			action_debug.append({})
		if _game and _game.has_method("env_get_action_stats"):
			action_stats.append(_game.env_get_action_stats(team))
		else:
			action_stats.append({})

	var done_info = _termination_engine.consume_done_info()
	var response_info = info.duplicate(true)
	response_info["reward_terms"] = reward_terms
	response_info["action_debug"] = action_debug
	response_info["action_stats"] = action_stats
	response_info["freeze_on_action_timeout_s"] = float(_ticks_per_step) / float(max(1, Engine.physics_ticks_per_second))
	response_info["action_timeout_freeze_active"] = _timeout_freeze_active
	response_info["action_timeout_freeze_count"] = _timeout_freeze_count
	response_info["waiting_for_step_sim_s"] = _waiting_for_step_sim_s
	if _game and _game.has_method("env_consume_spawn_queue_debug"):
		var queue_debug = _game.env_consume_spawn_queue_debug()
		if queue_debug is Dictionary:
			for key in queue_debug.keys():
				response_info[key] = queue_debug[key]

	return {
		"obs": obs,
		"action_mask": action_mask_flat,
		"reward": reward,
		"done": bool(done_info.get("done", false)),
		"truncation": bool(done_info.get("truncation", false)),
		"info": response_info,
	}
