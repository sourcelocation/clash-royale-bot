extends Node
class_name RewardEngine

var _game: Node = null


func bind_game(game: Node) -> void:
	_game = game


func consume_reward_payload(for_team: int) -> Dictionary:
	if not _game:
		return {"reward": 0.0, "terms": {}}
	var reward = _game.env_get_reward(for_team)
	var terms = {}
	if _game.has_method("env_consume_reward_terms"):
		terms = _game.env_consume_reward_terms(for_team)
	_game.env_clear_reward(for_team)
	return {"reward": reward, "terms": terms}
