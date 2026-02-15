extends Node
class_name ActionMasker

var _game: Node = null


func bind_game(game: Node) -> void:
	_game = game


func encode_flat(for_team: int) -> Array:
	if not _game:
		return []
	return _game.get_action_mask_flat(for_team)


func encode_branches(for_team: int) -> Dictionary:
	if not _game:
		return {}
	return _game.get_action_mask(for_team)
