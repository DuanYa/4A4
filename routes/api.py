"""
REST API 路由 - 房间管理
"""
from flask import Blueprint, jsonify, request
import os
import logging

api_bp = Blueprint('api', __name__)
logger = logging.getLogger('api')

# 房间管理器引用（由server.py注入）
room_manager = None


def init_api(rm):
    global room_manager
    room_manager = rm


@api_bp.route('/api/rooms', methods=['GET'])
def list_rooms():
    logger.info('REST list_rooms: room_count=%d', len(room_manager))
    rooms = []
    for rid, room in room_manager.items():
        rooms.append(room.get_room_state())
    return jsonify({'rooms': rooms})


@api_bp.route('/api/rooms', methods=['POST'])
def create_room():
    data = request.get_json() or {}
    room_id = data.get('room_id', '')
    if not room_id:
        import uuid
        room_id = str(uuid.uuid4())[:8]
    if room_id in room_manager:
        logger.info('REST create_room failed: room exists room_id=%s', room_id)
        return jsonify({'error': 'room_exists'}), 400
    from models.room import Room
    room_manager[room_id] = Room(room_id)
    logger.info('REST create_room success: room_id=%s', room_id)
    return jsonify(room_manager[room_id].get_room_state())


@api_bp.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    """列出可供AI选择的模型checkpoint文件"""
    root = os.path.dirname(os.path.dirname(__file__))
    ckpt_dir = os.path.join(root, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    result = []
    for name in sorted(os.listdir(ckpt_dir)):
        if name.endswith(('.pt', '.pth')):
            result.append(name)
    if 'default_rl.pt' not in result:
        result.insert(0, 'default_rl.pt')
    logger.info('REST list_checkpoints: count=%d files=%s', len(result), result)
    return jsonify({'checkpoints': result})


@api_bp.route('/api/rooms/<room_id>', methods=['GET'])
def get_room(room_id):
    if room_id not in room_manager:
        logger.info('REST get_room not_found: room_id=%s', room_id)
        return jsonify({'error': 'room_not_found'}), 404
    logger.info('REST get_room success: room_id=%s', room_id)
    return jsonify(room_manager[room_id].get_room_state())
