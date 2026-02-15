extends Entity
class_name Spell

signal resolved(team: int, hit_count: int)

var hit_count: int = 0
var _resolved_emitted: bool = false
var _hit_entity_ids: Dictionary = {}


func _ready():
	super._ready()


func register_hit(entity: Entity) -> void:
	if not entity or not is_instance_valid(entity):
		return
	var entity_id = int(entity.get_instance_id())
	if _hit_entity_ids.has(entity_id):
		return
	_hit_entity_ids[entity_id] = true
	hit_count += 1


func _emit_resolved_once() -> void:
	if _resolved_emitted:
		return
	_resolved_emitted = true
	resolved.emit(team, hit_count)


func _exit_tree() -> void:
	_emit_resolved_once()
