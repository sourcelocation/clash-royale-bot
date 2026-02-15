extends Troop


func _die() -> void:
	for area in $DeathArea.get_overlapping_areas():
		if area is Entity and area.team != team:
			area.update_health(-card.damage)
			area.stun_timer = max(area.stun_timer, 2.0)
	super._die()
