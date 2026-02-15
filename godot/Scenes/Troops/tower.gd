extends Building
class_name Tower

signal tower_damaged(team: int, damage: int)
signal tower_destroyed(team: int)

@export var manual_health: int = 500
@export var projectile_speed: float = 2.5
@export var projectile_cooldown: float = 0.8
@export var projectile_damage: int = 160
@export var activated: bool = true
@export var distance: float = 8.5
@onready var projectile_scene: PackedScene = preload("res://Scenes/Projectiles/tower_arrow.tscn")

var projectile_timer: float = 0.0
var destroyed: bool = false
var locked_target: Entity = null

func _ready() -> void:
	super._ready()
	health = manual_health
	update_health(0)
	

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if stun_timer == 0: projectile_timer = max(0, projectile_timer - delta)

	if projectile_timer == 0 and not destroyed and activated:
		if not locked_target or not is_instance_valid(locked_target):
			locked_target = get_closest_enemy_in_range()
		if locked_target:
			var locked_target_position = Vector2(locked_target.position.x, locked_target.position.z)
			var my_position_2d = Vector2(position.x, position.z)
			if my_position_2d.distance_to(locked_target_position) > distance:
				locked_target = null

		
		if locked_target and is_instance_valid(locked_target):
			var projectile = projectile_scene.instantiate() as Projectile
			projectile.position = position + Vector3(0, 0.2, 0)
			projectile.target = locked_target
			projectile.speed = projectile_speed
			projectile.damage = projectile_damage
			get_parent().add_child(projectile)
			projectile_timer = projectile_cooldown
	
		
func update_health(health_delta: int, _custom_max_health = null) -> void:
	super.update_health(health_delta, manual_health if manual_health != -1 else _custom_max_health)

	tower_damaged.emit(team, health_delta)

func _die() -> void:
	visible = false
	ui.visible = false
	for c in $Particles.get_children():
		if c is GPUParticles3D:
			c.restart()

	destroyed = true
	set_collision_layer(0)
	set_collision_mask(0)
	tower_destroyed.emit(team)

func get_closest_enemy_in_range() -> Entity:
	var enemies = get_node("../..").entities.filter(func(e):
		return e and (e is Building or e is Troop) and e.team != team
	)
	var closest_enemy: Entity = null
	var closest_distance = INF
	for enemy in enemies:
		var v2_1 = Vector2(position.x, position.z)
		var v2_2 = Vector2(enemy.position.x, enemy.position.z)
		var dist = v2_1.distance_to(v2_2)
		if dist < closest_distance and dist <= distance * ArenaConf.tile_size.x:
			closest_distance = dist
			closest_enemy = enemy
	return closest_enemy
