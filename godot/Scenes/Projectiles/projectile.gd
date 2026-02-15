extends Area3D
class_name Projectile

var target: Entity
var last_target_position: Vector3
var speed: float = 0.5
var damage: int = 10

func _ready(): pass

func _physics_process(delta: float) -> void:
	if target and is_instance_valid(target):
		last_target_position = target.position
	position = position.move_toward(last_target_position, speed * delta)
	var direction = (last_target_position - position).normalized()
	if direction.length() > 0:
		look_at(position + direction, Vector3.UP)
	else:
		queue_free()

	for area in get_overlapping_areas():
		if area is Entity and area != self and area == target:
			area.update_health(-damage)
			queue_free()

	if target == null or not is_instance_valid(target):
		if position.distance_to(last_target_position) < 0.1:
			queue_free()
