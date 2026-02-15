extends Node
class_name EnvTransportServer

var adapter: ArenaEnvAdapter
var env_id: int = 0
var port: int = 12000

var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
var _recv_buffer := ""
var _is_listening := false
var _is_configured := false
var _last_seq := -1
var _step_response_pending := false
var _pending_step_seq := -1


func configure(adapter_ref: ArenaEnvAdapter, p_port: int, p_env_id: int) -> void:
	adapter = adapter_ref
	port = p_port
	env_id = p_env_id
	_is_configured = true
	if is_inside_tree():
		_start_server()


func _ready() -> void:
	if _is_configured:
		_start_server()


func _process(_delta: float) -> void:
	if not _is_listening:
		return

	if not _peer and _server.is_connection_available():
		_peer = _server.take_connection()
		_peer.set_no_delay(true)
		print("[EnvTransportServer] Client connected on port %d" % port)

	if not _peer:
		return

	_peer.poll()
	if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_peer = null
		_recv_buffer = ""
		_last_seq = -1
		_step_response_pending = false
		_pending_step_seq = -1
		return

	var available = _peer.get_available_bytes()
	if available > 0:
		_recv_buffer += _peer.get_utf8_string(available)
		_process_buffer()

	if _step_response_pending and adapter and adapter.has_step_result():
		_send_response(
			{
				"type": "step",
				"env_id": env_id,
				"seq": _pending_step_seq,
				"data": adapter.take_step_result(),
			}
		)
		_step_response_pending = false
		_pending_step_seq = -1


func _start_server() -> void:
	if _is_listening:
		return
	var err = _server.listen(port, "127.0.0.1")
	if err != OK:
		push_error("[EnvTransportServer] Failed to listen on port %d (err=%d)" % [port, err])
		return
	_is_listening = true
	print("[EnvTransportServer] Listening on 127.0.0.1:%d env_id=%d" % [port, env_id])


func _process_buffer() -> void:
	while true:
		var newline_index = _recv_buffer.find("\n")
		if newline_index == -1:
			return

		var line = _recv_buffer.substr(0, newline_index).strip_edges()
		_recv_buffer = _recv_buffer.substr(newline_index + 1)
		if line.is_empty():
			continue

		var request = JSON.parse_string(line)
		if typeof(request) != TYPE_DICTIONARY:
			_send_response({"type": "error", "env_id": env_id, "seq": -1, "error": "invalid_json"})
			continue

		var response = _handle_request(request)
		if response != null:
			_send_response(response)


func _handle_request(request: Dictionary):
	var req_type = str(request.get("type", ""))
	var seq = int(request.get("seq", -1))
	if seq >= 0 and seq <= _last_seq:
		return {
			"type": "error",
			"env_id": env_id,
			"seq": seq,
			"error": "stale_or_reordered_seq",
			"last_seq": _last_seq,
		}
	_last_seq = max(_last_seq, seq)

	match req_type:
		"hello":
			return {
				"type": "hello",
				"env_id": env_id,
				"seq": seq,
				"protocol_version": adapter.PROTOCOL_VERSION if adapter else "unknown",
			}
		"spec":
			return {
				"type": "spec",
				"env_id": env_id,
				"seq": seq,
				"spec": adapter.get_spec() if adapter else {},
			}
		"reset":
			var seed_value = int(request.get("seed", -1))
			var options = request.get("options", {})
			if typeof(options) != TYPE_DICTIONARY:
				options = {}
			return {
				"type": "reset",
				"env_id": env_id,
				"seq": seq,
				"data": adapter.reset(seed_value, options) if adapter else {},
			}
		"step":
			if _step_response_pending:
				return {
					"type": "error",
					"env_id": env_id,
					"seq": seq,
					"error": "step_already_pending",
				}
			var actions = request.get("actions", [])
			if typeof(actions) != TYPE_ARRAY:
				return {
					"type": "error",
					"env_id": env_id,
					"seq": seq,
					"error": "actions_must_be_array",
				}
			if not adapter:
				return {
					"type": "step",
					"env_id": env_id,
					"seq": seq,
					"data": {},
				}
			var begin = adapter.begin_step(actions)
			if not bool(begin.get("accepted", false)):
				return {
					"type": "step",
					"env_id": env_id,
					"seq": seq,
					"data": begin.get("data", {}),
				}
			_step_response_pending = true
			_pending_step_seq = seq
			return null
		"ping":
			return {"type": "pong", "env_id": env_id, "seq": seq}
		"close":
			call_deferred("_shutdown")
			return {"type": "close_ack", "env_id": env_id, "seq": seq}
		_:
			return {
				"type": "error",
				"env_id": env_id,
				"seq": seq,
				"error": "unknown_request_type",
			}


func _send_response(response: Dictionary) -> void:
	if not _peer or _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return
	var payload = JSON.stringify(response) + "\n"
	_peer.put_data(payload.to_utf8_buffer())


func _shutdown() -> void:
	if _peer:
		_peer.disconnect_from_host()
	if _is_listening:
		_server.stop()
	_last_seq = -1
	_step_response_pending = false
	_pending_step_seq = -1
	get_tree().quit()
