extends Node

@onready var arena = get_parent()
@export var team: int = 1

func _physics_process(_delta):
	# Bot behavior is now owned by game.gd via team controller config.
	pass
