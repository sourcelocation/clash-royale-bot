extends Area3D
class_name Entity

signal died

var health = 100
var card: Card
var dead = false

var stun_timer: float = 0.0

var begin_attack_timer: Timer
var end_attack_timer: Timer
var attacking: bool = false
var can_attack: bool = true
var deployment_locked: bool = false
var range_area: Area3D
var optimize = false

var has_set_stylebox = false
var ui
@export var team: int = 0
@export var visual: Node3D
var health_bar: ProgressBar
@onready var health_fill_blue: StyleBoxFlat = preload("res://Scenes/Troops/health-fill-blue.tres")
@onready var health_fill_red: StyleBoxFlat = preload("res://Scenes/Troops/health-fill-red.tres")

func _ready():
	if has_node("UI"):
		ui = $UI
		health_bar = $UI/HealthBar
	if card:
		begin_attack_timer = Timer.new()
		begin_attack_timer.wait_time = card.attack_swing
		begin_attack_timer.one_shot = true
		begin_attack_timer.timeout.connect(_on_begin_attack_timeout)
		add_child(begin_attack_timer)
		end_attack_timer = Timer.new()
		end_attack_timer.wait_time = card.attack_end
		end_attack_timer.one_shot = true
		end_attack_timer.timeout.connect(_on_end_attack_timeout)
		add_child(end_attack_timer)

	visual.visible = !optimize
	if has_node("RangeArea"):
		range_area = $RangeArea
	if optimize:
		if has_node("UI"):
			$UI.visible = false

func set_card(c: Card, _team: int = 0) -> void:
	card = c
	team = _team
	health = card.health

	if health_bar:
		health_bar.max_value = health
		health_bar.value = health

func get_speed() -> float:
	return card.speed if card else 0.0

func _physics_process(_delta):
	_update_ui()

	stun_timer = max(0, stun_timer - _delta)

	if deployment_locked:
		cancel_attack()
		return

	if can_attack and range_area:
		var colliding_areas = range_area.get_overlapping_areas()
		for area in colliding_areas:
			if area is Entity and area.team != team:
				if card.target_type == Card.TargetType.BUILDINGS and not (area is Building):
					continue
				if not attacking and begin_attack_timer.is_stopped() and stun_timer <= 0.0:
					attacking = true
					can_attack = false
					begin_attack_timer.start()

func update_health(health_delta: int, custom_max_health = null) -> void:
	health += health_delta
	if health_bar and not optimize:
		health_bar.max_value = custom_max_health if custom_max_health else card.health
		health_bar.value = health
		if not has_set_stylebox:
			health_bar.add_theme_stylebox_override("fill", health_fill_blue if team == 0 else health_fill_red)
			has_set_stylebox = true
	if health_delta < 0:
		if has_node("AnimationPlayer"): $AnimationPlayer.play("DamageHit")
	if not dead and health <= 0:
		dead = true
		_die()


func _die() -> void:
	queue_free()

	died.emit()

func _update_ui():
	if ui and is_inside_tree():
		var camera = get_viewport().get_camera_3d()
		if camera:
			var screen_pos = camera.unproject_position(global_position)
			
			ui.global_position = screen_pos + Vector2(0, -50) - ui.size / 2
	update_health(0)

func _get_speed() -> float:
	return card.speed if card else 0.0


func _on_begin_attack_timeout() -> void:
	var colliding_areas = range_area.get_overlapping_areas()
	for area in colliding_areas:
		if area is Entity and area.team != team:
			_deal_damage_to_target(area)
			break

	end_attack_timer.start()

func _on_end_attack_timeout() -> void:
	attacking = false
	can_attack = true


func _deal_damage_to_target(target: Entity) -> void:
	if target and is_instance_valid(target):
		target.update_health(-card.damage)


func cancel_attack() -> void:
	if attacking:
		attacking = false
		can_attack = true
		if begin_attack_timer and begin_attack_timer.is_stopped() == false:
			begin_attack_timer.stop()
	if end_attack_timer and end_attack_timer.is_stopped() == false:
		end_attack_timer.stop()


func set_deployment_lock() -> void:
	deployment_locked = true
	cancel_attack()


func clear_deployment_lock() -> void:
	deployment_locked = false
