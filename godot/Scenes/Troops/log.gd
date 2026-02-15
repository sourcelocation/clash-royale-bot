extends Spell

var affected_entities: Array = []
var time = 0.0

func _physics_process(delta: float) -> void:
	time += delta
	if time >= 2.0:
		queue_free()
		return
	var colliding_areas = range_area.get_overlapping_areas()
	for area in colliding_areas:
		if area is Entity and area.team != team and not affected_entities.has(area):
			affected_entities.append(area)
			register_hit(area)
			if area is Tower:
				area.update_health(-card.tower_damage)
			else:
				area.update_health(-card.damage)
			area.stun_timer = max(area.stun_timer, 0.4)

			if area is Troop:
				(area as Troop).cancel_attack()

	position.z += delta * card.speed * (-1 if team == 0 else 1)
	rotation.x += delta * 6.0 * (-1 if team == 0 else 1)
	rotation.z = sin(time * 10.0) * 0.05
