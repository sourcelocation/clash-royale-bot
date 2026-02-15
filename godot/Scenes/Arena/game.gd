extends Node3D
const CARDS = preload("res://Globals/cards.tres")
const USABLE_CARD_COUNT := 8
const HAND_SIZE := 4
const ACTION_ORDER := ["wait", "card_selection", "position_region", "position_cell"]
# const BUILDING_EXCLUSION_RADIUS_TILES := 1.2
const MAX_ENTITIES_PER_TEAM_OBS := 8
const TOWER_HP_DELTA_REWARD_SCALE := 0.6
const TOWER_HP_DELTA_REWARD_CLIP := 0.05
const TERMINAL_REWARD := 3.0
const HOG_CARD_INDEX := 3
const CHEAP_CARD_MAX_COST := 2
const CHEAP_CYCLE_PENALTY := 0.003
const HOG_STOP_REWARD := 0.04
const BUILDING_BACKLINE_PENALTY_BASE := 1.0
const BUILDING_BACKLINE_MAX_ROW := 6
const SPELL_HIT_REWARD := 0.01
const SPELL_MISS_PENALTY := 0.015
const SPELL_HIT_REWARD_MAX_TARGETS := 4
const ELIXIR_OVERCAP_DELAY_S := 2.0
const ELIXIR_OVERCAP_TICK_S := 1.0
const ELIXIR_OVERCAP_PENALTY := 0.002
const PLACEMENT_REGION_COLS := 3
const PLACEMENT_REGION_ROWS := 3
const QUEUE_HZ := 10.0
const QUEUE_DT_S := 1.0 / QUEUE_HZ
const SPAWN_MIN_DELAY_S := 1.0
const DEPLOY_MIN_DELAY_S := 0.5

var elixir_1 = 8.0
var elixir_2 = 8.0

var time_since_tower_damaged = 0.0

var deck_1: Array = []
var deck_2: Array = []

var entities: Array = []

var selected_card_index = -1

var optimize = false

var time_since_tower_reward = 0.0
var accumulated_damage_1 = 0
var accumulated_damage_2 = 0

@export var base_seconds_per_elixir = 2.8
@export var max_elixir = 10.0
@export var range_of_troops = 8.0
@export var debug_mouse_grid: bool = true
@export var human_input_enabled: bool = true
@export var show_queue_indicators: bool = true

@onready var ai1: Node3D = $AI1

@onready var bottom_panel: Panel = %BottomPanel
@onready var arena_static_body: StaticBody3D = $ArenaStaticBody
@onready var arena_collision_shape: CollisionShape3D = $ArenaStaticBody/CollisionShape3D
@onready var switch_team_button: Button = $UI/SwitchTeamButton

var physics_frame_count = 0
var start_time = Time.get_ticks_msec()
var env_episode_done := false
var env_episode_truncated := false
var env_external_reset_control := false
var env_reset_requested := false
var env_rewards := [0.0, 0.0]
var env_reward_terms := [ {}, {}]
var env_training_mode := false
var team_controllers: Array[String] = ["human", "external"]
var _last_reset_seed: int = -1
var _last_tower_health := [0.0, 0.0]
var _initial_tower_health := [1.0, 1.0]
var _overcap_time := [0.0, 0.0]
var _overcap_penalty_accum := [0.0, 0.0]
var _env_last_action_debug: Array = [ {}, {}]
var _env_action_stats: Array = [ {}, {}]
var _queue_accum_s := 0.0
var _queue_time_s := 0.0
var _next_spawn_id := 1
var _pending_spawns: Array = []
var _queue_events_started: Array = []
var _queue_events_activated: Array = []

func _physics_process(_delta: float) -> void:
	physics_frame_count += 1

	elixir_1 += _delta / base_seconds_per_elixir
	elixir_2 += _delta / base_seconds_per_elixir
	elixir_1 = clampf(elixir_1, 0, max_elixir)
	elixir_2 = clampf(elixir_2, 0, max_elixir)
	bottom_panel.update(elixir_1)
	_process_queue_scheduler(_delta)

	for entity in entities:
		if entity and entity is Troop:
			var target_pos = get_target_position(entity.position, entity.team, entity.card)
			entity.set_target_position(target_pos)

	if ai1.needs_reset:
		_request_episode_end(true, true)

	time_since_tower_damaged += _delta

	# _collect_elixir_overcap_penalties(_delta)

	time_since_tower_reward += _delta
	if time_since_tower_reward >= 0.2:
		time_since_tower_reward = 0.0
		_collect_tower_rewards()

func _collect_tower_rewards() -> void:
	var current_team_0 = _sum_visible_tower_health(0)
	var current_team_1 = _sum_visible_tower_health(1)
	var damage_to_team_0 = max(0.0, _last_tower_health[0] - current_team_0)
	var damage_to_team_1 = max(0.0, _last_tower_health[1] - current_team_1)

	if damage_to_team_1 > 0.0:
		var dealt_ratio = damage_to_team_1 / max(1.0, _initial_tower_health[1])
		var dealt = clampf(dealt_ratio * TOWER_HP_DELTA_REWARD_SCALE, 0.0, TOWER_HP_DELTA_REWARD_CLIP)
		_add_reward(0, dealt, "tower_damage_dealt")
		_add_reward(1, -dealt, "tower_damage_taken")
	if damage_to_team_0 > 0.0:
		var taken_ratio = damage_to_team_0 / max(1.0, _initial_tower_health[0])
		var taken = clampf(taken_ratio * TOWER_HP_DELTA_REWARD_SCALE, 0.0, TOWER_HP_DELTA_REWARD_CLIP)
		_add_reward(0, -taken, "tower_damage_taken")
		_add_reward(1, taken, "tower_damage_dealt")

	_last_tower_health[0] = current_team_0
	_last_tower_health[1] = current_team_1

# func _collect_elixir_overcap_penalties(delta: float) -> void:
# 	for team in range(2):
# 		var current_elixir = elixir_1 if team == 0 else elixir_2
# 		if current_elixir >= max_elixir - 1e-6:
# 			_overcap_time[team] += delta
# 			if _overcap_time[team] > ELIXIR_OVERCAP_DELAY_S:
# 				_overcap_penalty_accum[team] += delta
# 				while _overcap_penalty_accum[team] >= ELIXIR_OVERCAP_TICK_S:
# 					_overcap_penalty_accum[team] -= ELIXIR_OVERCAP_TICK_S
# 					_add_reward(team, -ELIXIR_OVERCAP_PENALTY, "elixir_overcap_sustained")
# 		else:
# 			_overcap_time[team] = 0.0
# 			_overcap_penalty_accum[team] = 0.0

func _ready() -> void:
	randomize()
	if optimize:
		$UI.visible = false
	_reset_game()
	_update_hand()
	

	visible = !optimize
	if not optimize:
		$ReflectionProbe.visible = true
		#$LightmapGI.visible = true
	
	bottom_panel.card_pressed.connect(_card_selected)
	_ensure_selected_action_label()
	_refresh_selected_action_label()
	var arena_shape = arena_collision_shape.shape as BoxShape3D
	ArenaConf.arena_size = arena_shape.size
	ArenaConf.tile_size = Vector2(ArenaConf.arena_size.x / ArenaConf.grid_tiles_count.x, ArenaConf.arena_size.z / ArenaConf.grid_tiles_count.y)

	entities += $Towers.get_children()

	for tower in entities:
		if not tower or not is_instance_valid(tower): continue
		tower.optimize = optimize
		if tower is Tower:
			tower.tower_damaged.connect(
				func(team: int, damage: int) -> void:
					if team == 0:
						accumulated_damage_1 -= damage
					else:
						accumulated_damage_2 -= damage

					time_since_tower_damaged = 0.0
			)
			tower.tower_destroyed.connect(
				func(team: int) -> void:
					if team == 0:
						_add_reward(0, -TERMINAL_REWARD, "tower_destroyed")
						_add_reward(1, TERMINAL_REWARD, "tower_kill")
					else:
						_add_reward(0, TERMINAL_REWARD, "tower_kill")
						_add_reward(1, -TERMINAL_REWARD, "tower_destroyed")
					_request_episode_end(true, false)
			)

func _reset_game(seed_value: int = -1):
	if seed_value >= 0:
		_last_reset_seed = seed_value
		seed(seed_value)
	_collect_tower_rewards()
	ai1.done = true
	ai1.reset()
	
	for entity in entities:
		if not entity or not is_instance_valid(entity): continue
		entity.dead = false
		if entity is not Tower:
			entity.queue_free()
		else:
			entity.visible = true
			entity.ui.visible = true
			entity.destroyed = false
			entity.set_collision_layer(1)
			entity.set_collision_mask(1)
			entity.health = entity.manual_health
			entity.update_health(0)

	entities = $Towers.get_children()

	elixir_1 = 8.0
	elixir_2 = 8.0
	time_since_tower_damaged = 0.0
	time_since_tower_reward = 0.0
	accumulated_damage_1 = 0
	accumulated_damage_2 = 0
	deck_1 = range(USABLE_CARD_COUNT)
	deck_2 = range(USABLE_CARD_COUNT)
	deck_1.shuffle()
	deck_2.shuffle()
	selected_card_index = -1
	_update_hand()

	var fps = float(physics_frame_count) / (Time.get_ticks_msec() - start_time) * 1000.0
	print("FPS: ", fps)
	print("Actual Speedup: ", fps / 60.0)
	physics_frame_count = 0
	start_time = Time.get_ticks_msec()
	ai1.done = false
	ai1.needs_reset = false
	env_episode_done = false
	env_episode_truncated = false
	env_reset_requested = false
	env_rewards = [0.0, 0.0]
	env_reward_terms = [ {}, {}]
	_initial_tower_health = [_sum_visible_tower_health(0), _sum_visible_tower_health(1)]
	_last_tower_health = _initial_tower_health.duplicate()
	_overcap_time = [0.0, 0.0]
	_overcap_penalty_accum = [0.0, 0.0]
	_env_last_action_debug = [ {}, {}]
	_env_action_stats = [
		{
			"decisions": 0,
			"applied": 0,
			"wait": 0,
			"rejections": {},
		},
		{
			"decisions": 0,
			"applied": 0,
			"wait": 0,
			"rejections": {},
		},
	]
	_queue_accum_s = 0.0
	_queue_time_s = 0.0
	_next_spawn_id = 1
	_clear_pending_spawn_indicators()
	_pending_spawns = []
	_queue_events_started = []
	_queue_events_activated = []
	_refresh_selected_action_label()

func _add_reward(team: int, delta: float, term: String = "misc") -> void:
	if team < 0 or team >= env_rewards.size():
		return
	env_rewards[team] += delta
	if term.is_empty():
		term = "misc"
	var terms_for_team = env_reward_terms[team]
	terms_for_team[term] = float(terms_for_team.get(term, 0.0)) + delta
	env_reward_terms[team] = terms_for_team

func _request_episode_end(done: bool, truncated: bool) -> void:
	env_episode_done = env_episode_done or done
	env_episode_truncated = env_episode_truncated or truncated
	env_reset_requested = true
	ai1.done = env_episode_done
	ai1.needs_reset = true
	if not env_external_reset_control:
		_reset_game()

func env_is_external_reset_control() -> bool:
	return env_external_reset_control

func env_set_external_reset_control(enabled: bool) -> void:
	env_external_reset_control = enabled

func env_set_training_mode(enabled: bool) -> void:
	env_training_mode = enabled
	if enabled:
		human_input_enabled = false
		debug_mouse_grid = false
		_clear_pending_spawn_indicators()

func env_set_matchup(team0_controller: String, team1_controller: String) -> void:
	env_set_team_controller(0, team0_controller)
	env_set_team_controller(1, team1_controller)

func env_set_team_controller(team: int, controller: String) -> void:
	if team < 0 or team >= team_controllers.size():
		return
	var normalized = controller.to_lower()
	if normalized == "selfplay" or normalized == "selfplay_pool":
		normalized = "external"
	if normalized != "human" and normalized != "external":
		push_warning("Invalid team controller '%s' for team %d; keeping previous value." % [controller, team])
		return
	team_controllers[team] = normalized
	if team == 0:
		human_input_enabled = (normalized == "human") and not env_training_mode

func env_get_team_controller(team: int) -> String:
	if team < 0 or team >= team_controllers.size():
		return "external"
	return team_controllers[team]

func env_get_action_space() -> Dictionary:
	var region_count = _placement_region_count()
	var cell_count = _placement_cells_per_region()
	return {
		"wait": {"size": 2, "action_type": "discrete"},
		"card_selection": {"size": USABLE_CARD_COUNT, "action_type": "discrete"},
		"position_region": {"size": region_count, "action_type": "discrete"},
		"position_cell": {"size": cell_count, "action_type": "discrete"},
	}

func env_get_placement_schema() -> Dictionary:
	var x_size = _action_grid_width()
	var y_size = _action_grid_height()
	var region_w = _region_width()
	var region_h = _region_height()
	return {
		"grid_width": x_size,
		"grid_height": y_size,
		"position_count": x_size * y_size,
		"region_cols": PLACEMENT_REGION_COLS,
		"region_rows": PLACEMENT_REGION_ROWS,
		"region_count": _placement_region_count(),
		"region_cell_width": region_w,
		"region_cell_height": region_h,
		"region_cell_count": region_w * region_h,
	}

func env_get_reward(for_team: int) -> float:
	if for_team < 0 or for_team >= env_rewards.size():
		return 0.0
	return env_rewards[for_team]

func env_clear_reward(for_team: int) -> void:
	if for_team < 0 or for_team >= env_rewards.size():
		return
	env_rewards[for_team] = 0.0

func env_consume_reward_terms(for_team: int) -> Dictionary:
	if for_team < 0 or for_team >= env_reward_terms.size():
		return {}
	var terms = env_reward_terms[for_team].duplicate(true)
	env_reward_terms[for_team] = {}
	return terms

func env_consume_done_info() -> Dictionary:
	var done_info = {
		"done": env_episode_done,
		"truncation": env_episode_truncated,
		"reset_requested": env_reset_requested,
	}
	return done_info

func _update_hand():
	var cards_deck_1 = deck_1.slice(0, 4).map(
		func(x): return CARDS.cards[int(x)]
	)
	bottom_panel.create_cards(cards_deck_1)

func _card_selected(i):
	selected_card_index = i

var debug_snap_marker: MeshInstance3D
var debug_raw_marker: MeshInstance3D
var debug_label: Label
var selected_action_label: Label

func _input(event: InputEvent) -> void:
	if not human_input_enabled:
		return
	_ensure_mouse_debug_nodes()

	if event is InputEventMouseMotion:
		var hover_hit = _mouse_to_arena_world()
		if hover_hit != null:
			var snapped_pos = snapped_position(hover_hit)
			snapped_pos.y = hover_hit.y + 0.01
			_update_mouse_debug(hover_hit, snapped_pos)

	if event is InputEventMouseButton and event.pressed \
		and event.button_index == MOUSE_BUTTON_LEFT:
		if selected_card_index != -1:
			var card_i = int(deck_1[selected_card_index])
			var click_hit = _mouse_to_arena_world()
			if click_hit != null:
				var snapped_pos = snapped_position(click_hit)
				var team = 1 if switch_team_button.button_pressed else 0
				var action_grid = _world_to_action_grid(team, snapped_pos)
				if player_play_card(team, card_i, action_grid):
					selected_card_index = -1
					

func _spawn_entity(card_index: int, _position: Vector3, team: int) -> Array:
	var card = CARDS.cards[card_index]
	var amount = card.amount
	var created: Array = []

	for i in range(amount):
		var new_instance = card.scene.instantiate()
		new_instance.position = _position
		# new_instance.tile_size = ArenaConf.tile_size
		new_instance.set_card(card, team)

		var collision_shape = new_instance.get_node_or_null("CollisionShape3D")
		if collision_shape:
			var height_offset
			if collision_shape.shape is CylinderShape3D:
				height_offset = (collision_shape.shape as CylinderShape3D).height / 2.0
			elif collision_shape.shape is BoxShape3D:
				height_offset = (collision_shape.shape as BoxShape3D).size.y / 2.0
			elif collision_shape.shape is SphereShape3D:
				height_offset = (collision_shape.shape as SphereShape3D).radius
			else:
				height_offset = 0.0
			new_instance.position.y += height_offset

		new_instance.optimize = optimize
		add_child(new_instance)
		if new_instance.has_signal("died"):
			new_instance.died.connect(_on_env_entity_died.bind(team, card_index))
		if new_instance is Spell:
			(new_instance as Spell).resolved.connect(_on_spell_resolved)
		if new_instance is Troop:
			var target_pos = get_target_position(new_instance.position, new_instance.team, card)
			new_instance.set_target_position(target_pos)
		entities.append(new_instance)
		created.append(new_instance)
	return created


func env_reset(seed_value: int = -1) -> void:
	_reset_game(seed_value)


func env_step(team: int, action: Dictionary) -> bool:
	if env_get_team_controller(team) != "external":
		_record_env_action_debug(team, action, Vector2.ZERO, false, "non_external_controller")
		return false
	if env_episode_done or env_episode_truncated:
		_record_env_action_debug(team, action, Vector2.ZERO, false, "episode_done")
		return false

	var wait = int(action.get("wait", 0))
	var card_selection = int(action.get("card_selection", -1))
	var position_region = int(action.get("position_region", 0))
	var position_cell = int(action.get("position_cell", 0))
	var action_grid = _region_cell_to_action_grid(position_region, position_cell)
	if wait == 1:
		_record_env_action_debug(team, action, action_grid, false, "wait")
		return false

	var play_result = _play_card_with_reason(team, card_selection, action_grid)
	var applied = bool(play_result.get("applied", false))
	var reason = str(play_result.get("reason", "unknown"))
	var debug_meta = play_result.get("debug_meta", {})
	if typeof(debug_meta) != TYPE_DICTIONARY:
		debug_meta = {}
	_record_env_action_debug(team, action, action_grid, applied, reason, debug_meta)
	return applied


func env_get_obs(for_team: int) -> Dictionary:
	return {
		"obs": get_obs_space(for_team),
		"action_mask": get_action_mask_flat(for_team),
		"position_masks_for_all_cards": get_position_masks_for_all_cards(for_team),
	}


func get_action_mask(for_team: int) -> Dictionary:
	var deck = deck_1 if for_team == 0 else deck_2
	var current_elixir = elixir_1 if for_team == 0 else elixir_2
	var hand = deck.slice(0, HAND_SIZE)
	var region_count = _placement_region_count()
	var cell_count = _placement_cells_per_region()

	var card_selection_mask: Array = []
	var has_playable_card = false
	var position_region_mask = _filled_mask(region_count, 0.0)
	var position_cell_mask = _filled_mask(cell_count, 0.0)
	for card_index in range(USABLE_CARD_COUNT):
		var card = CARDS.cards[card_index]
		var can_play = hand.has(card_index) and card.cost <= current_elixir \
			and _has_any_legal_position(for_team, card)
		card_selection_mask.append(1.0 if can_play else 0.0)
		if can_play:
			has_playable_card = true
	var wait_mask = [1.0 if has_playable_card else 0.0, 1.0]

	# Region/cell branches are conditioned on sampled card in policy code.
	# Keep permissive base masks here.
	if has_playable_card:
		for i in range(position_region_mask.size()):
			position_region_mask[i] = 1.0
		for i in range(position_cell_mask.size()):
			position_cell_mask[i] = 1.0

	return {
		"wait": wait_mask,
		"card_selection": card_selection_mask,
		"position_region": position_region_mask,
		"position_cell": position_cell_mask,
	}


func get_action_mask_flat(for_team: int) -> Array:
	var mask_by_branch = get_action_mask(for_team)
	var flat_mask: Array = []
	for key in ACTION_ORDER:
		flat_mask.append_array(mask_by_branch[key])
	return flat_mask


func _filled_mask(size: int, value: float) -> Array:
	var values: Array = []
	values.resize(size)
	for i in range(size):
		values[i] = value
	return values

func _is_action_grid_in_bounds(action_grid: Vector2) -> bool:
	var x_size = _action_grid_width()
	var y_size = _action_grid_height()
	return action_grid.x >= 0 and action_grid.x < x_size \
		and action_grid.y >= 0 and action_grid.y < y_size

func _action_grid_to_position_index(action_grid: Vector2) -> int:
	var x_size = _action_grid_width()
	return int(action_grid.y) * x_size + int(action_grid.x)

func _action_grid_width() -> int:
	return int(ArenaConf.grid_tiles_count.x)

func _action_grid_height() -> int:
	return int(ArenaConf.grid_tiles_count.y / 2.0 - 1.0)

func _region_width() -> int:
	return int(ceil(float(_action_grid_width()) / float(max(1, PLACEMENT_REGION_COLS))))

func _region_height() -> int:
	return int(ceil(float(_action_grid_height()) / float(max(1, PLACEMENT_REGION_ROWS))))

func _placement_region_count() -> int:
	return max(1, PLACEMENT_REGION_COLS * PLACEMENT_REGION_ROWS)

func _placement_cells_per_region() -> int:
	return max(1, _region_width() * _region_height())

func _region_index_to_origin(region_index: int) -> Vector2:
	var cols = max(1, PLACEMENT_REGION_COLS)
	var idx = clampi(region_index, 0, _placement_region_count() - 1)
	var region_x = idx % cols
	var region_y = int(idx / cols)
	return Vector2(region_x * _region_width(), region_y * _region_height())

func _region_cell_to_action_grid(region_index: int, cell_index: int) -> Vector2:
	var origin = _region_index_to_origin(region_index)
	var region_w = _region_width()
	var region_h = _region_height()
	var max_cell = max(1, region_w * region_h) - 1
	var local_idx = clampi(cell_index, 0, max_cell)
	var local_x = local_idx % region_w
	var local_y = int(local_idx / region_w)
	return Vector2(origin.x + local_x, origin.y + local_y)

func _action_grid_to_world(team: int, action_grid: Vector2) -> Vector3:
	var internal_grid = _team_action_grid_to_internal(team, action_grid)
	return grid_position_to_world(internal_grid)

func _team_action_grid_to_internal(team: int, action_grid: Vector2) -> Vector2:
	var internal_grid = action_grid
	if team == 0:
		internal_grid.y = ArenaConf.grid_tiles_count.y - 1 - internal_grid.y
		internal_grid.x = ArenaConf.grid_tiles_count.x - 1 - internal_grid.x
	return internal_grid

func _world_to_action_grid(team: int, world_pos: Vector3) -> Vector2:
	var internal_grid = world_position_to_grid(world_pos)
	if team == 0:
		return Vector2(
			ArenaConf.grid_tiles_count.x - 1 - internal_grid.x,
			ArenaConf.grid_tiles_count.y - 1 - internal_grid.y
		)
	return internal_grid

func _is_world_position_playable(card: Card, world_pos: Vector3, team: int) -> bool:
	if not is_position_in_arena(world_pos):
		return false

	# var building_exclusion = BUILDING_EXCLUSION_RADIUS_TILES * ArenaConf.tile_size.x

	# if card.type == Card.CardType.BUILDING:
	# 	for entity in entities:
	# 		if not entity or not is_instance_valid(entity):
	# 			continue
	# 		if entity is not Building and entity is not Tower:
	# 			continue
	# 		if entity.team != team:
	# 			continue
	# 		if not entity.visible:
	# 			continue
	# 		if world_pos.distance_to(entity.position) < building_exclusion:
	# 			return false

	return true

func _compute_valid_position_mask(team: int, card: Card) -> Array:
	var x_size = _action_grid_width()
	var y_size = _action_grid_height()
	var valid_positions = _filled_mask(x_size * y_size, 0.0)

	for x in range(x_size):
		for y in range(y_size):
			var action_grid = Vector2(x, y)
			if not _is_action_grid_in_bounds(action_grid):
				continue
			var world_pos = _action_grid_to_world(team, action_grid)
			if _is_world_position_playable(card, world_pos, team):
				var index = _action_grid_to_position_index(action_grid)
				valid_positions[index] = 1.0

	return valid_positions

func get_position_mask_for_card(for_team: int, card_selection: int) -> Array:
	var total = _action_grid_width() * _action_grid_height()
	var empty_mask = _filled_mask(total, 0.0)

	if card_selection < 0 or card_selection >= USABLE_CARD_COUNT:
		return empty_mask

	var deck = deck_1 if for_team == 0 else deck_2
	var hand = deck.slice(0, HAND_SIZE)
	if not hand.has(card_selection):
		return empty_mask

	var current_elixir = elixir_1 if for_team == 0 else elixir_2
	var card = CARDS.cards[card_selection]
	if card.cost > current_elixir:
		return empty_mask

	return _compute_valid_position_mask(for_team, card)

func get_position_masks_for_all_cards(for_team: int) -> Array:
	var masks: Array = []
	for card_selection in range(USABLE_CARD_COUNT):
		masks.append(get_position_mask_for_card(for_team, card_selection))
	return masks

func env_consume_action_debug(for_team: int) -> Dictionary:
	if for_team < 0 or for_team >= _env_last_action_debug.size():
		return {}
	var payload = _env_last_action_debug[for_team].duplicate(true)
	_env_last_action_debug[for_team] = {}
	return payload

func env_get_action_stats(for_team: int) -> Dictionary:
	if for_team < 0 or for_team >= _env_action_stats.size():
		return {}
	return _env_action_stats[for_team].duplicate(true)

func env_consume_spawn_queue_debug() -> Dictionary:
	var pending_by_team = [0, 0]
	for pending in _pending_spawns:
		if typeof(pending) != TYPE_DICTIONARY:
			continue
		var team = int(pending.get("team", -1))
		if team >= 0 and team < pending_by_team.size():
			pending_by_team[team] = int(pending_by_team[team]) + 1

	var payload = {
		"pending_spawns_total": int(_pending_spawns.size()),
		"pending_spawns_team": pending_by_team,
		"spawns_started_this_step": _queue_events_started.duplicate(true),
		"spawns_activated_this_step": _queue_events_activated.duplicate(true),
		"queue_hz": QUEUE_HZ,
		"queue_dt_s": QUEUE_DT_S,
		"spawn_min_delay_s": SPAWN_MIN_DELAY_S,
		"deploy_min_delay_s": DEPLOY_MIN_DELAY_S,
	}
	_queue_events_started = []
	_queue_events_activated = []
	return payload

func _record_env_action_debug(
	team: int,
	action: Dictionary,
	action_grid: Vector2,
	applied: bool,
	reason: String,
	meta: Dictionary = {}
) -> void:
	if team < 0 or team >= _env_last_action_debug.size():
		return
	var world_position = _action_grid_to_world(team, action_grid)
	_env_last_action_debug[team] = {
		"team": team,
		"wait": int(action.get("wait", 1)),
		"card_selection": int(action.get("card_selection", -1)),
		"position_region": int(action.get("position_region", 0)),
		"position_cell": int(action.get("position_cell", 0)),
		"grid_x": int(action_grid.x),
		"grid_y": int(action_grid.y),
		"world_x": float(world_position.x),
		"world_z": float(world_position.z),
		"applied": applied,
		"reason": reason,
		"queued": bool(meta.get("queued", false)),
		"spawn_id": int(meta.get("spawn_id", -1)),
		"queue_hz": float(QUEUE_HZ),
		"spawn_min_delay_s": float(SPAWN_MIN_DELAY_S),
		"deploy_min_delay_s": float(DEPLOY_MIN_DELAY_S),
	}
	var stats = _env_action_stats[team]
	stats["decisions"] = int(stats.get("decisions", 0)) + 1
	if int(action.get("wait", 0)) == 1:
		stats["wait"] = int(stats.get("wait", 0)) + 1
	if applied:
		stats["applied"] = int(stats.get("applied", 0)) + 1
	elif reason != "wait":
		var rejections: Dictionary = stats.get("rejections", {})
		rejections[reason] = int(rejections.get(reason, 0)) + 1
		stats["rejections"] = rejections
	_env_action_stats[team] = stats
	_refresh_selected_action_label()

func _has_any_legal_position(team: int, card: Card) -> bool:
	var valid_positions = _compute_valid_position_mask(team, card)
	for v in valid_positions:
		if v > 0.0:
			return true
	return false

func snapped_position(pos: Vector3) -> Vector3:
	var grid_pos = world_position_to_grid(pos)
	var world_pos = grid_position_to_world(grid_pos)
	return Vector3(world_pos.x, pos.y, world_pos.z)

func world_position_to_grid(world_pos: Vector3) -> Vector2:
	var x_size = int(ArenaConf.grid_tiles_count.x)
	var y_size = int(ArenaConf.grid_tiles_count.y)
	var half_extents = _get_arena_half_extents()
	var tile_size = _get_arena_tile_size()
	var local_pos = arena_collision_shape.to_local(world_pos)

	var gx = int(floor((local_pos.x + half_extents.x) / tile_size.x))
	var gy = int(floor((local_pos.z + half_extents.y) / tile_size.y))
	gx = clampi(gx, 0, x_size - 1)
	gy = clampi(gy, 0, y_size - 1)
	return Vector2(gx, gy)

func grid_position_to_world(grid_pos: Vector2) -> Vector3:
	var half_extents = _get_arena_half_extents()
	var tile_size = _get_arena_tile_size()
	var local_x = - half_extents.x + (grid_pos.x + 0.5) * tile_size.x
	var local_z = - half_extents.y + (grid_pos.y + 0.5) * tile_size.y
	return arena_collision_shape.to_global(Vector3(local_x, 0.0, local_z))

func is_position_in_arena(pos: Vector3) -> bool:
	var half_extents = _get_arena_half_extents()
	var local_pos = arena_collision_shape.to_local(pos)
	return local_pos.x >= -half_extents.x and local_pos.x <= half_extents.x \
		and local_pos.z >= -half_extents.y and local_pos.z <= half_extents.y

func _mouse_to_arena_world() -> Variant:
	var mouse_pos = get_viewport().get_mouse_position()
	var ray_origin = $Camera3D.project_ray_origin(mouse_pos)
	var ray_end = ray_origin + $Camera3D.project_ray_normal(mouse_pos) * 1000.0
	var query = PhysicsRayQueryParameters3D.create(ray_origin, ray_end)
	query.collision_mask = arena_static_body.collision_layer
	query.collide_with_areas = false
	query.collide_with_bodies = true
	var world = get_world_3d()
	if world == null:
		return null
	var hit = world.direct_space_state.intersect_ray(query)
	if hit.is_empty():
		return null
	if hit.get("collider", null) != arena_static_body:
		return null

	var world_hit: Vector3 = hit["position"]
	if not is_position_in_arena(world_hit):
		return null
	return world_hit

func _get_arena_half_extents() -> Vector2:
	var shape = arena_collision_shape.shape as BoxShape3D
	if shape == null:
		return Vector2(0.5, 0.5)
	return Vector2(shape.size.x * 0.5, shape.size.z * 0.5)

func _get_arena_tile_size() -> Vector2:
	var half_extents = _get_arena_half_extents()
	var full_size = half_extents * 2.0
	return Vector2(
		full_size.x / max(1.0, ArenaConf.grid_tiles_count.x),
		full_size.y / max(1.0, ArenaConf.grid_tiles_count.y)
	)

func _ensure_selected_action_label() -> void:
	if selected_action_label != null:
		return
	selected_action_label = Label.new()
	selected_action_label.position = Vector2(16, 240)
	selected_action_label.size = Vector2(900, 72)
	selected_action_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	selected_action_label.vertical_alignment = VERTICAL_ALIGNMENT_TOP
	selected_action_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	selected_action_label.modulate = Color(1.0, 0.95, 0.8, 1.0)
	$UI.add_child(selected_action_label)

func _format_action_debug_line(team: int) -> String:
	var payload = _env_last_action_debug[team]
	if typeof(payload) != TYPE_DICTIONARY or payload.is_empty():
		return "T%d: -" % team
	var wait = int(payload.get("wait", 1))
	if wait == 1:
		return "T%d: WAIT reason=%s" % [team, str(payload.get("reason", "unknown"))]
	return "T%d: card=%d region=%d cell=%d grid=(%d,%d) applied=%s reason=%s" % [
		team,
		int(payload.get("card_selection", -1)),
		int(payload.get("position_region", 0)),
		int(payload.get("position_cell", 0)),
		int(payload.get("grid_x", 0)),
		int(payload.get("grid_y", 0)),
		str(bool(payload.get("applied", false))),
		str(payload.get("reason", "unknown")),
	]

func _refresh_selected_action_label() -> void:
	_ensure_selected_action_label()
	if selected_action_label == null:
		return
	selected_action_label.visible = true
	selected_action_label.text = "Selected Action\n%s | %s" % [
		_format_action_debug_line(0),
		_format_action_debug_line(1),
	]

func _ensure_mouse_debug_nodes() -> void:
	if debug_snap_marker == null:
		debug_snap_marker = _create_debug_marker(Color(0.2, 1.0, 0.2, 1.0))
		add_child(debug_snap_marker)
	if debug_raw_marker == null:
		debug_raw_marker = _create_debug_marker(Color(1.0, 0.3, 0.3, 1.0))
		add_child(debug_raw_marker)
	if debug_label == null:
		debug_label = Label.new()
		debug_label.position = Vector2(16, 16)
		debug_label.size = Vector2(520, 220)
		debug_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		debug_label.vertical_alignment = VERTICAL_ALIGNMENT_TOP
		debug_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		$UI.add_child(debug_label)

	if debug_mouse_grid:
		debug_snap_marker.visible = true
		debug_raw_marker.visible = true
		debug_label.visible = true
	else:
		debug_snap_marker.visible = false
		debug_raw_marker.visible = false
		debug_label.visible = false

func _create_debug_marker(color: Color) -> MeshInstance3D:
	var marker = MeshInstance3D.new()
	var mesh = SphereMesh.new()
	mesh.radius = 0.02
	mesh.height = 0.04
	marker.mesh = mesh
	var material = StandardMaterial3D.new()
	material.albedo_color = color
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	marker.material_override = material
	return marker

func _update_mouse_debug(raw_hit: Vector3, snapped_pos: Vector3) -> void:
	if not debug_mouse_grid:
		return

	var grid = world_position_to_grid(raw_hit)
	var delta = raw_hit - snapped_pos
	var tile_size = _get_arena_tile_size()
	var delta_tiles = Vector2(
		delta.x / max(0.00001, tile_size.x),
		delta.z / max(0.00001, tile_size.y)
	)
	var center_x = arena_static_body.global_position.x
	var center_z = arena_static_body.global_position.z
	var mouse_pos = get_viewport().get_mouse_position()
	var raw_screen = $Camera3D.unproject_position(raw_hit)
	var screen_delta = raw_screen - mouse_pos

	debug_raw_marker.position = raw_hit
	debug_snap_marker.position = snapped_pos
	debug_label.text = "Mouse Grid Debug\n" \
		+"raw_hit: (%.4f, %.4f, %.4f)\n" % [raw_hit.x, raw_hit.y, raw_hit.z] \
		+"snapped: (%.4f, %.4f, %.4f)\n" % [snapped_pos.x, snapped_pos.y, snapped_pos.z] \
		+"grid: (%d, %d)\n" % [int(grid.x), int(grid.y)] \
		+"delta world: (%.4f, %.4f)\n" % [delta.x, delta.z] \
		+"delta tiles: (%.4f, %.4f)\n" % [delta_tiles.x, delta_tiles.y] \
		+"screen delta px: (%.2f, %.2f)\n" % [screen_delta.x, screen_delta.y] \
		+"arena center: (%.4f, %.4f)\n" % [center_x, center_z] \
		+"tile_size: (%.4f, %.4f)" % [tile_size.x, tile_size.y]

func get_target_position(pos: Vector3, team: int, card: Card) -> Vector3:
	var enemy_in_range = get_enemy_in_range(pos, team, card)
	if enemy_in_range:
		return enemy_in_range.global_position
	return Vector3(0, pos.y, (ArenaConf.arena_size.z / 2 + 1) * (-1 if team == 0 else 1))

func get_enemy_in_range(pos: Vector3, team: int, card: Card) -> Entity:
	var closest_entity = null
	var closest_distance = INF
	for entity in entities:
		if not entity or not is_instance_valid(entity): continue
		if entity is not Troop and entity is not Building:
			continue
		if not entity.visible:
			continue
		if card.target_type == Card.TargetType.BUILDINGS and not (entity is Building):
			continue
		if entity.team != team:
			var distance = pos.distance_to(entity.position)
			if distance < closest_distance and (distance <= range_of_troops * ArenaConf.tile_size.x or entity is Tower):
				closest_distance = distance
				closest_entity = entity
	return closest_entity


func _play_card_with_reason(team: int, card_index: int, _position: Vector2) -> Dictionary:
	if card_index < 0 or card_index >= USABLE_CARD_COUNT:
		return {"applied": false, "reason": "invalid_card"}
	if not _is_action_grid_in_bounds(_position):
		return {"applied": false, "reason": "out_of_bounds"}

	var card = CARDS.cards[card_index]
	var hand = deck_1.slice(0, HAND_SIZE) if team == 0 else deck_2.slice(0, HAND_SIZE)
	var current_elixir = elixir_1 if team == 0 else elixir_2
	var hog_cost = CARDS.cards[HOG_CARD_INDEX].cost
	var should_penalize_cheap_cycle_play = hand.find(HOG_CARD_INDEX) != -1 \
		and current_elixir < hog_cost \
		and card_index != HOG_CARD_INDEX \
		and card.cost <= CHEAP_CARD_MAX_COST
	if team == 0:
		if card.cost > elixir_1:
			return {"applied": false, "reason": "insufficient_elixir"}
		if deck_1.slice(0, 4).find(card_index) == -1:
			return {"applied": false, "reason": "card_not_in_hand"}
	else:
		if card.cost > elixir_2:
			return {"applied": false, "reason": "insufficient_elixir"}
		if deck_2.slice(0, 4).find(card_index) == -1:
			return {"applied": false, "reason": "card_not_in_hand"}
		# if not is_position_in_arena(Vector3(position.x, 0, position.y)):
	# 	return
	var world_position = _action_grid_to_world(team, _position)
	if not _is_world_position_playable(card, world_position, team):
		return {"applied": false, "reason": "illegal_position"}

	if team == 0:
		elixir_1 -= card.cost
		var index_in_hand = deck_1.slice(0, 4).find(card_index)
		deck_1.remove_at(index_in_hand)
		deck_1.append(card_index)
		var new_card = deck_1[3]
		deck_1.remove_at(3)
		deck_1.insert(index_in_hand, new_card)
		_update_hand()
	else:
		elixir_2 -= card.cost
		var index_in_hand = deck_2.slice(0, 4).find(card_index)
		deck_2.remove_at(index_in_hand)
		deck_2.append(card_index)
		var new_card = deck_2[3]
		deck_2.remove_at(3)
		deck_2.insert(index_in_hand, new_card)

	if card.type == Card.CardType.BUILDING:
		var team_row = int(_position.y)
		if team_row <= BUILDING_BACKLINE_MAX_ROW:
			var depth = BUILDING_BACKLINE_MAX_ROW - team_row + 1
			var placement_penalty = float(depth) * BUILDING_BACKLINE_PENALTY_BASE
			_add_reward(team, -placement_penalty, "building_backline")

	if should_penalize_cheap_cycle_play:
		_add_reward(team, -CHEAP_CYCLE_PENALTY, "cheap_cycle_play")

	var spawn_id = _queue_card_spawn(team, card_index, world_position)
	return {
		"applied": true,
		"reason": "queued",
		"spawn_id": spawn_id,
		"debug_meta": {
			"queued": true,
			"spawn_id": spawn_id,
		},
	}

func player_play_card(team: int, card_index: int, _position: Vector2) -> bool:
	return bool(_play_card_with_reason(team, card_index, _position).get("applied", false))


func _on_spell_resolved(caster_team: int, hit_count: int) -> void:
	if env_episode_done or env_episode_truncated:
		return
	if caster_team < 0 or caster_team >= 2:
		return
	var enemy_team = 1 - caster_team
	if hit_count <= 0:
		_add_reward(caster_team, -SPELL_MISS_PENALTY, "spell_miss")
		_add_reward(enemy_team, SPELL_MISS_PENALTY, "enemy_spell_miss")
		return
	var clamped_hits = min(hit_count, SPELL_HIT_REWARD_MAX_TARGETS)
	var bonus = float(clamped_hits) * SPELL_HIT_REWARD
	_add_reward(caster_team, bonus, "spell_hit")
	_add_reward(enemy_team, -bonus, "enemy_spell_hit_taken")


func _on_env_entity_died(entity_team: int, card_index: int) -> void:
	if env_episode_done or env_episode_truncated:
		return
	if card_index != HOG_CARD_INDEX:
		return
	if entity_team < 0 or entity_team >= 2:
		return
	var stopper_team = 1 - entity_team
	_add_reward(stopper_team, HOG_STOP_REWARD, "hog_stopped")
	_add_reward(entity_team, -HOG_STOP_REWARD, "hog_lost")

func _queue_card_spawn(team: int, card_index: int, world_position: Vector3) -> int:
	var spawn_id = _next_spawn_id
	_next_spawn_id += 1
	var indicator: Node3D = null
	if _queue_indicators_enabled():
		indicator = _create_queue_indicator(team, card_index, world_position)
	_pending_spawns.append(
		{
			"spawn_id": spawn_id,
			"team": team,
			"card_index": card_index,
			"world_position": world_position,
			"spawn_at_s": _queue_time_s + SPAWN_MIN_DELAY_S,
			"active_at_s": - 1.0,
			"state": "queued",
			"entities": [],
			"indicator": indicator,
		}
	)
	return spawn_id

func _process_queue_scheduler(delta: float) -> void:
	_queue_accum_s += delta
	while _queue_accum_s >= QUEUE_DT_S:
		_queue_accum_s -= QUEUE_DT_S
		_queue_time_s += QUEUE_DT_S
		_process_queue_tick()

func _process_queue_tick() -> void:
	var remove_indices: Array = []
	for i in range(_pending_spawns.size()):
		var pending = _pending_spawns[i]
		if typeof(pending) != TYPE_DICTIONARY:
			remove_indices.append(i)
			continue
		var state = str(pending.get("state", "queued"))
		if state == "queued":
			var spawn_at_s = float(pending.get("spawn_at_s", _queue_time_s))
			_update_pending_indicator(pending, max(0.0, spawn_at_s - _queue_time_s), false)
			if _queue_time_s < spawn_at_s:
				_pending_spawns[i] = pending
				continue
			var card_index = int(pending.get("card_index", -1))
			var team = int(pending.get("team", 0))
			if card_index < 0 or card_index >= CARDS.cards.size():
				_destroy_pending_indicator(pending)
				remove_indices.append(i)
				continue
			var card = CARDS.cards[card_index]
			var world_position: Vector3 = pending.get("world_position", Vector3.ZERO)
			var created = _spawn_entity(card_index, world_position, team)
			_queue_events_started.append(
				{
					"spawn_id": int(pending.get("spawn_id", -1)),
					"team": team,
					"card_index": card_index,
				}
			)
			if card.type == Card.CardType.TROOP or card.type == Card.CardType.BUILDING:
				for entity in created:
					if entity and is_instance_valid(entity) and entity.has_method("set_deployment_lock"):
						entity.set_deployment_lock()
				pending["state"] = "deploying"
				pending["active_at_s"] = _queue_time_s + DEPLOY_MIN_DELAY_S
				pending["entities"] = created
				_update_pending_indicator(pending, DEPLOY_MIN_DELAY_S, true)
				_pending_spawns[i] = pending
			else:
				_destroy_pending_indicator(pending)
				_queue_events_activated.append(
					{
						"spawn_id": int(pending.get("spawn_id", -1)),
						"team": team,
						"card_index": card_index,
					}
				)
				remove_indices.append(i)
		elif state == "deploying":
			var active_at_s = float(pending.get("active_at_s", _queue_time_s))
			_update_pending_indicator(pending, max(0.0, active_at_s - _queue_time_s), true)
			if _queue_time_s < active_at_s:
				_pending_spawns[i] = pending
				continue
			var deployed_entities = pending.get("entities", [])
			if deployed_entities is Array:
				for entity in deployed_entities:
					if entity and is_instance_valid(entity) and entity.has_method("clear_deployment_lock"):
						entity.clear_deployment_lock()
			_destroy_pending_indicator(pending)
			_queue_events_activated.append(
				{
					"spawn_id": int(pending.get("spawn_id", -1)),
					"team": int(pending.get("team", 0)),
					"card_index": int(pending.get("card_index", -1)),
				}
			)
			remove_indices.append(i)
		else:
			_destroy_pending_indicator(pending)
			remove_indices.append(i)

	while not remove_indices.is_empty():
		var idx = int(remove_indices.pop_back())
		if idx >= 0 and idx < _pending_spawns.size():
			_pending_spawns.remove_at(idx)

func _queue_indicators_enabled() -> bool:
	return show_queue_indicators and not optimize and not env_training_mode

func _create_queue_indicator(team: int, card_index: int, world_position: Vector3) -> Node3D:
	var root = Node3D.new()
	root.position = Vector3(world_position.x, world_position.y + 0.04, world_position.z)

	var card = CARDS.cards[card_index]
	var radius = 1.0 * ArenaConf.tile_size.x

	var ring_mesh = MeshInstance3D.new()
	ring_mesh.name = "Ring"
	var cylinder = CylinderMesh.new()
	cylinder.top_radius = radius
	cylinder.bottom_radius = radius
	cylinder.height = 0.02
	ring_mesh.mesh = cylinder
	var ring_mat = StandardMaterial3D.new()
	ring_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	ring_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	ring_mat.albedo_color = Color(1.0, 0.85, 0.15, 0.45) if team == 0 else Color(0.9, 0.35, 0.25, 0.45)
	ring_mesh.material_override = ring_mat
	root.add_child(ring_mesh)

	# var label = Label3D.new()
	# label.name = "Countdown"
	# label.position = Vector3(0.0, 0.45, 0.0)
	# label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	# label.modulate = Color(1.0, 0.95, 0.85, 1.0)
	# label.font_size = 32
	# label.outline_size = 8
	# label.outline_modulate = Color(0.0, 0.0, 0.0, 0.75)
	# label.text = "1.0s"
	# root.add_child(label)

	add_child(root)
	return root

func _update_pending_indicator(pending: Dictionary, remaining_s: float, deploying: bool) -> void:
	if not pending.has("indicator"):
		return
	var indicator = pending.get("indicator")
	if indicator == null or not (indicator is Node3D) or not is_instance_valid(indicator):
		return

	var ring = indicator.get_node_or_null("Ring")
	var label = indicator.get_node_or_null("Countdown")
	if ring and ring is MeshInstance3D:
		var material = ring.material_override
		if material and material is StandardMaterial3D:
			var team = int(pending.get("team", 0))
			if deploying:
				material.albedo_color = Color(0.25, 0.85, 1.0, 0.5) if team == 0 else Color(0.75, 0.55, 1.0, 0.5)
			else:
				material.albedo_color = Color(1.0, 0.85, 0.15, 0.45) if team == 0 else Color(0.9, 0.35, 0.25, 0.45)
	if label and label is Label3D:
		var prefix = "Deploy" if deploying else "Spawn"
		label.text = "%s %.1fs" % [prefix, max(0.0, remaining_s)]

func _destroy_pending_indicator(pending: Dictionary) -> void:
	if not pending.has("indicator"):
		return
	var indicator = pending.get("indicator")
	if indicator and indicator is Node3D and is_instance_valid(indicator):
		indicator.queue_free()

func _clear_pending_spawn_indicators() -> void:
	for pending in _pending_spawns:
		if pending is Dictionary:
			_destroy_pending_indicator(pending)

func get_obs_space(for_team: int) -> Array:
	var should_flip = for_team == 1
	var res = []
	var arena_size = ArenaConf.grid_tiles_count
	var card_norm_denom = max(1.0, float(USABLE_CARD_COUNT - 1))
	var team_self = 0 if not should_flip else 1
	var team_enemy = 1 if not should_flip else 0

	# Elixir (normalized to [0, 1])
	res.append(min(elixir_1, 10.0) / 10.0 if not should_flip else min(elixir_2, 10.0) / 10.0)
	res.append(min(elixir_2, 10.0) / 10.0 if not should_flip else min(elixir_1, 10.0) / 10.0)
	res.append(clampf(time_since_tower_damaged / 15.0, 0.0, 1.0))
	res.append(_count_alive_towers(team_self) / 3.0)
	res.append(_count_alive_towers(team_enemy) / 3.0)
	res.append(_count_playable_cards_in_hand(team_self) / float(HAND_SIZE))
	res.append(_count_playable_cards_in_hand(team_enemy) / float(HAND_SIZE))

	# Hand cards (normalized indices)
	var deck_1_slice = deck_1.slice(0, HAND_SIZE)
	var deck_2_slice = deck_2.slice(0, HAND_SIZE)
	for i in range(HAND_SIZE):
		if should_flip:
			res.append(deck_2_slice[i] / card_norm_denom if deck_2_slice[i] >= 0 else -1.0)
			res.append(deck_1_slice[i] / card_norm_denom if deck_1_slice[i] >= 0 else -1.0)
		else:
			res.append(deck_1_slice[i] / card_norm_denom if deck_1_slice[i] >= 0 else -1.0)
			res.append(deck_2_slice[i] / card_norm_denom if deck_2_slice[i] >= 0 else -1.0)

	# Entity slots: fixed-size padded schema per team.
	var team_a_entities = entities.filter(func(e): return e and e is Entity and e.team == (0 if not should_flip else 1))
	var team_b_entities = entities.filter(func(e): return e and e is Entity and e.team == (1 if not should_flip else 0))

	for team_entities in [team_a_entities, team_b_entities]:
		team_entities.sort_custom(
			func(a: Entity, b: Entity) -> bool:
				var az = a.position.z if not should_flip else -a.position.z
				var bz = b.position.z if not should_flip else -b.position.z
				if not is_equal_approx(az, bz):
					return az < bz
				var ax = a.position.x if not should_flip else -a.position.x
				var bx = b.position.x if not should_flip else -b.position.x
				if not is_equal_approx(ax, bx):
					return ax < bx
				return a.get_instance_id() < b.get_instance_id()
		)
		for i in range(MAX_ENTITIES_PER_TEAM_OBS):
			if i < team_entities.size():
				var entity = team_entities[i]
				var card_index = CARDS.cards.find(entity.card) if entity.card in CARDS.cards else -1
				var normalized_card_index = card_index if card_index >= 0 and card_index < USABLE_CARD_COUNT else -1
				# Feature order: exists, x, z, card_idx, hp_frac, is_building.
				var x = entity.position.x if not should_flip else -entity.position.x
				var z = entity.position.z if not should_flip else -entity.position.z
				res.append(1.0)
				res.append((x + arena_size.x / 2.0) / arena_size.x)
				res.append((z + arena_size.y / 2.0) / arena_size.y)
				res.append(
					normalized_card_index / card_norm_denom if normalized_card_index >= 0 else -1.0
				)
				var max_health = float(entity.card.health if entity.card else entity.manual_health)
				res.append(float(entity.health) / max(1e-6, max_health))
				res.append(1.0 if (entity is Building or entity is Tower) else 0.0)
			else:
				res.append(0.0)
				res.append(0.0)
				res.append(0.0)
				res.append(-1.0)
				res.append(0.0)
				res.append(0.0)
	for i in range(res.size()):
		if not is_finite(float(res[i])):
			res[i] = 0.0
	return res

func _sum_visible_tower_health(team: int) -> float:
	var total = 0.0
	for entity in entities:
		if not entity or not is_instance_valid(entity):
			continue
		if entity is Tower and entity.team == team and entity.visible:
			total += float(entity.health)
	return total

func _count_alive_towers(team: int) -> float:
	var count = 0.0
	for entity in entities:
		if not entity or not is_instance_valid(entity):
			continue
		if entity is Tower and entity.team == team and entity.visible:
			count += 1.0
	return count

func _count_playable_cards_in_hand(team: int) -> float:
	var deck = deck_1 if team == 0 else deck_2
	var current_elixir = elixir_1 if team == 0 else elixir_2
	var hand = deck.slice(0, HAND_SIZE)
	var count = 0.0
	for card_index in hand:
		if card_index < 0 or card_index >= USABLE_CARD_COUNT:
			continue
		var card = CARDS.cards[card_index]
		if card.cost <= current_elixir and _has_any_legal_position(team, card):
			count += 1.0
	return count
