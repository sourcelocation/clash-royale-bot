extends Entity
class_name Building

var health_label: Label

var damage_to_deal: float = 0.0

func _ready() -> void:
	super._ready()
	update_health(0)

	if $UI and $UI.has_node("HealthLabel"):
		health_label = $UI/HealthLabel
	
func update_health(health_delta: int, _custom_max_health = null) -> void:
	super.update_health(health_delta, _custom_max_health)
	if health_label:
		health_label.text = str(health)

func _die() -> void:
	queue_free()

func _physics_process(delta: float) -> void:
	super._physics_process(delta)

	if card:
		if card.decay_time > 0.0:
			damage_to_deal += (card.health / card.decay_time) * delta

	if damage_to_deal > 1.0:
		update_health(-int(damage_to_deal))
		damage_to_deal -= int(damage_to_deal)
