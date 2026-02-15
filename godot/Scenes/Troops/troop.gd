extends Entity
class_name Troop

@onready var navigation_agent: NavigationAgent3D = $NavigationAgent3D
@onready var collision_shape: CollisionShape3D = $CollisionShape3D


func _ready():
	super._ready()
	navigation_agent.target_desired_distance = 0.1 # card.attack_range * tile_size.x

	range_area.get_node("CollisionShape3D").shape.radius = card.attack_range * ArenaConf.tile_size.x

func set_card(c: Card, _team: int = 0) -> void:
	super.set_card(c, _team)
	

func set_target_position(target: Vector3) -> void:
	if not attacking and not deployment_locked:
		navigation_agent.set_target_position(target)

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	

	_attempt_move(delta)

func _attempt_move(delta: float) -> void:
	# Move towards target
	if deployment_locked:
		return
	if not attacking and stun_timer <= 0.0:
		if not navigation_agent.is_navigation_finished():
			var next_path_position_global = navigation_agent.get_next_path_position()
			# print(next_path_position_global)
			var direction = (next_path_position_global - global_position).normalized()
			var velocity = direction * card.speed * delta
			global_position.x += velocity.x
			global_position.z += velocity.z
			rotation = Vector3(0, atan2(direction.x, direction.z), 0)

	# Simple collision avoidance with other entities
	var colliding_areas = get_overlapping_areas()
	for area in colliding_areas:
		if area is Entity:
			var speed_diff = clampf(area._get_speed() - _get_speed(), 0, 100)
			
			var diff = (area.position - position).normalized() * card.speed * delta * (1.0 + speed_diff * 1.5)
			position.x -= diff.x
			position.z -= diff.z
