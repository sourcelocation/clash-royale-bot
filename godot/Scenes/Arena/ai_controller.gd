extends Node3D
class_name ArenaAIController

@export var heuristic: String = "model"

@export var team: int = 0
@onready var arena = get_parent()
var reward: float = 0.0
var done: bool = false
var needs_reset: bool = false
var n_steps: int = 0
var player_node: Node = null

func _ready() -> void:
	randomize()

func init(player: Node) -> void:
	player_node = player

func reset() -> void:
	reward = 0.0
	n_steps = 0
	needs_reset = false

func get_obs() -> Dictionary:
	return arena.env_get_obs(team)

func get_obs_space() -> Dictionary:
	var obs_dict = get_obs()
	return {
		"obs": {"size": [len(obs_dict["obs"])], "space": "box"},
		"action_mask": {"size": [len(obs_dict["action_mask"])], "space": "box"},
	}

func get_reward() -> float:
	return reward

func _physics_process(_delta):
	if done:
		return

func get_action_space() -> Dictionary:
	var arena_size: Vector2 = ArenaConf.grid_tiles_count
	var res = {
		"card_selection": {"size": 8, "action_type": "discrete"},
		"position_x": {"size": int(arena_size.x), "action_type": "discrete"},
		"position_y": {"size": int(arena_size.y / 2.0 - 1.0), "action_type": "discrete"},
		"wait": {"size": 2, "action_type": "discrete"},
	}
	return res

func set_action(action) -> void:	
	arena.env_step(team, action)
