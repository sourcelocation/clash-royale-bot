extends Node
class_name ObservationEncoder

var _game: Node = null


func bind_game(game: Node) -> void:
	_game = game


func encode(for_team: int) -> Dictionary:
	if not _game:
		return {"vector": [], "spatial": [], "meta": {"team": for_team}}

	var obs_dict = _game.env_get_obs(for_team)
	return {
		"vector": obs_dict.get("obs", []),
		"position_masks_for_all_cards": obs_dict.get("position_masks_for_all_cards", []),
		# Spatial channels are intentionally added as the next stage of the env encoder.
		"spatial": [],
		"meta": {"team": for_team},
	}
