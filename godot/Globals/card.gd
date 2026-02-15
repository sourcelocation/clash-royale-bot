extends Resource
class_name Card

enum CardType {
    TROOP,
    BUILDING,
    SPELL
}

enum TargetType {
    GROUND,
    AIR,
    BOTH,
    BUILDINGS
}

@export var name: String
@export var type: CardType
@export var texture: Texture2D
@export var cost: int
@export var amount: int = 1
@export var speed: float = 1.2
@export var health: int = 100
@export var damage: int = 10
@export var tower_damage: int = 5
@export var attack_swing: float = 0.3
@export var attack_end: float = 0.8
@export var attack_range: float = 1.0
@export var target_type: TargetType = TargetType.BOTH
@export var decay_time: float = 30.0
@export var scene: PackedScene
