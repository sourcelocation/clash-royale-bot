extends Troop

func _on_begin_attack_timeout() -> void:
	super._on_begin_attack_timeout()
	queue_free()

func _deal_damage_to_target(target: Entity) -> void:
	super._deal_damage_to_target(target)

	# AoE damage around the target
	var aoe_radius = 1.5 * ArenaConf.tile_size.x
	var entities = get_parent().entities.filter(func(e):
		return e and e is Entity and e.team != team and e != target
	)
	for entity in entities:
		if not entity or not is_instance_valid(entity): continue
		var distance = target.position.distance_to(entity.position)
		if distance <= aoe_radius:
			entity.update_health(-card.damage)
			entity.stun_timer = max(entity.stun_timer, 1.2)
	target.stun_timer = max(target.stun_timer, 1.2)
