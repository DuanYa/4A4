"""
四幺四(4A4)扑克游戏服务器入口
Flask + SocketIO 复用端口，同时提供REST API和WebSocket
"""
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import logging
from routes.api import api_bp, init_api
from routes.ws import init_ws

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
)
logger = logging.getLogger('server')

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'si-yao-si-4a4-secret'

socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

room_manager = {}

init_api(room_manager)
init_ws(socketio, room_manager)
app.register_blueprint(api_bp)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/mobile')
def mobile_index():
    return send_from_directory('static/mobile', 'index.html')


@app.route('/mobile/<path:path>')
def mobile_files(path):
    return send_from_directory('static/mobile', path)


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    logger.info('=== 四幺四(4A4) 扑克游戏服务器 ===')
    logger.info('访问 http://localhost:5000 开始游戏')
    logger.info('桌面版: http://localhost:5000')
    logger.info('移动版: http://localhost:5000/mobile')
    socketio.run(app, host='0.0.0.0', port=5000,
                 debug=True, allow_unsafe_werkzeug=True)
