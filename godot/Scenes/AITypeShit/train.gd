extends Node3D

enum LaunchMode {
	HUMAN_VS_BOT,
	BOT_VS_BOT,
}

@onready var game_scene = preload("res://Scenes/Arena/game.tscn")
@onready var adapter_script = preload("res://Scenes/Arena/env/arena_env_adapter.gd")
@onready var server_script = preload("res://Scenes/Arena/env/transport_server.gd")
@export var n: int = 1

var _port: int = -1
var _env_id: int = 0
var _has_cli_args: bool = false
var _engine_time_scale: float = 1.0
var _physics_tps: int = 60


func _parse_args() -> void:
	var args = OS.get_cmdline_user_args()
	print("args ", args)
	_has_cli_args = args.size() > 0
	if args.size() == 0:
		print("No command line arguments provided, using defaults.")
		return

	for arg in args:
		if arg.begins_with("--n="):
			n = int(arg.substr(4, arg.length()))
		elif arg.begins_with("--port="):
			_port = int(arg.substr(7, arg.length()))
		elif arg.begins_with("--env_id="):
			_env_id = int(arg.substr(9, arg.length()))
		elif arg.begins_with("--engine_time_scale="):
			_engine_time_scale = float(arg.substr(20, arg.length()))
		elif arg.begins_with("--physics_tps="):
			_physics_tps = int(arg.substr(14, arg.length()))


func _apply_engine_speed() -> void:
	_engine_time_scale = max(_engine_time_scale, 0.01)
	_physics_tps = maxi(_physics_tps, 1)
	Engine.time_scale = _engine_time_scale
	Engine.physics_ticks_per_second = _physics_tps
	Engine.max_fps = 0
	print(
		"[Train] Engine speed config: time_scale=%s physics_tps=%s max_fps=%s"
		% [str(_engine_time_scale), str(_physics_tps), str(Engine.max_fps)]
	)


func _ready() -> void:
	randomize()
	_parse_args()
	_apply_engine_speed()

	# Script/CLI and headless launches should never block on UI.
	if _has_cli_args or DisplayServer.get_name() == "headless":
		_start_games(LaunchMode.HUMAN_VS_BOT)
		return

	_show_mode_menu()


func _show_mode_menu() -> void:
	var layer = CanvasLayer.new()
	add_child(layer)

	var root = Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	layer.add_child(root)

	var panel = PanelContainer.new()
	panel.custom_minimum_size = Vector2(420, 220)
	panel.anchor_left = 0.5
	panel.anchor_top = 0.5
	panel.anchor_right = 0.5
	panel.anchor_bottom = 0.5
	panel.offset_left = -210
	panel.offset_top = -110
	panel.offset_right = 210
	panel.offset_bottom = 110
	root.add_child(panel)

	var layout = VBoxContainer.new()
	layout.add_theme_constant_override("separation", 12)
	panel.add_child(layout)

	var title = Label.new()
	title.text = "Choose Game Mode"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	layout.add_child(title)

	var human_vs_bot = Button.new()
	human_vs_bot.text = "Human vs Bot"
	human_vs_bot.pressed.connect(
		func() -> void:
			layer.queue_free()
			_start_games(LaunchMode.HUMAN_VS_BOT)
	)
	layout.add_child(human_vs_bot)

	var bot_vs_bot = Button.new()
	bot_vs_bot.text = "Bot vs Bot"
	bot_vs_bot.pressed.connect(
		func() -> void:
			layer.queue_free()
			_start_games(LaunchMode.BOT_VS_BOT)
	)
	layout.add_child(bot_vs_bot)


func _start_games(mode: LaunchMode) -> void:
	print("Number of games to instantiate: ", n)
	var first_game: Node = null
	var is_headless = OS.has_feature("headless") or DisplayServer.get_name() == "headless"
	for i in range(n):
		var game_instance = game_scene.instantiate()
		game_instance.name = "Game_%d" % i

		# Position games in a grid layout with sqrt(n) per row.
		var row = int(i / ceil(sqrt(n)))
		var col = i % int(ceil(sqrt(n)))
		var spacing = 5.0
		game_instance.position = Vector3(col * spacing, 0, row * spacing)
		game_instance.optimize = is_headless

		if mode == LaunchMode.BOT_VS_BOT:
			game_instance.env_set_matchup("external", "external")
		else:
			game_instance.env_set_matchup("human", "external")

		print("Starting game instance: ", game_instance.name, " at position ", game_instance.position)
		add_child(game_instance)
		if i == 0:
			first_game = game_instance

	if _port > 0 and first_game:
		var adapter = adapter_script.new()
		add_child(adapter)
		adapter.bind_game(first_game)

		var server = server_script.new()
		add_child(server)
		server.configure(adapter, _port, _env_id)
