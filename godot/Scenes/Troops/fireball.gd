extends Spell

func _ready() -> void:
	super._ready()
	range_area.get_node("CollisionShape3D").shape.radius = card.attack_range * ArenaConf.tile_size.x

	await get_tree().create_timer(1.0).timeout

	for area in range_area.get_overlapping_areas():
		if area is Entity and area.team != team:
			register_hit(area)
			if area is Tower:
				area.update_health(-card.tower_damage)
			else:
				area.update_health(-card.damage)
			area.stun_timer = max(area.stun_timer, 0.4)
	queue_free()
