extends Node
class_name TerminationEngine

var _game: Node = null


func bind_game(game: Node) -> void:
	_game = game


func consume_done_info() -> Dictionary:
	if not _game:
		return {"done": false, "truncation": false}
	return _game.env_consume_done_info()
