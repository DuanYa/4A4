"""
SQLite persistence for WeChat users, room membership, snapshots and events.
"""
import hashlib
import json
import os
import secrets
import sqlite3
import time
import urllib.parse
import urllib.request

DB_PATH = os.environ.get(
    'FOURA4_DB_PATH',
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', '4a4.sqlite3')
)


def _ts():
    return int(time.time())


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS wechat_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                openid TEXT NOT NULL UNIQUE,
                unionid TEXT,
                session_key TEXT,
                nickname TEXT,
                avatar_url TEXT,
                anonymous_id TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_login_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES wechat_users(id)
            );

            CREATE TABLE IF NOT EXISTS room_members (
                room_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                seat INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                is_host INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'online',
                socket_sid TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                PRIMARY KEY(room_id, user_id),
                UNIQUE(room_id, seat)
            );

            CREATE TABLE IF NOT EXISTS room_snapshots (
                room_id TEXT PRIMARY KEY,
                room_state_json TEXT NOT NULL,
                game_states_json TEXT,
                round_result_json TEXT,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS game_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL,
                user_id INTEGER,
                seat INTEGER,
                event TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_room_members_user
                ON room_members(user_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_game_events_room
                ON game_events(room_id, id);
            """
        )


def _wechat_code_to_session(code):
    appid = os.environ.get('WECHAT_APPID')
    secret = os.environ.get('WECHAT_SECRET')
    if not appid or not secret or not code:
        return None
    params = urllib.parse.urlencode({
        'appid': appid,
        'secret': secret,
        'js_code': code,
        'grant_type': 'authorization_code',
    })
    url = 'https://api.weixin.qq.com/sns/jscode2session?' + params
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    if data.get('errcode'):
        raise RuntimeError(data.get('errmsg') or 'wechat_login_failed')
    return data


def login_wechat(code='', anonymous_id='', nickname='', avatar_url=''):
    session = _wechat_code_to_session(code)
    if session and session.get('openid'):
        openid = session['openid']
        unionid = session.get('unionid')
        session_key = session.get('session_key')
    else:
        seed = anonymous_id or code or secrets.token_urlsafe(16)
        openid = 'dev_' + hashlib.sha256(seed.encode('utf-8')).hexdigest()[:32]
        unionid = None
        session_key = None

    t = _ts()
    with _connect() as conn:
        row = conn.execute(
            'SELECT id FROM wechat_users WHERE openid=?',
            (openid,)
        ).fetchone()
        if row:
            user_id = row['id']
            conn.execute(
                """
                UPDATE wechat_users
                   SET unionid=?, session_key=?, nickname=?, avatar_url=?,
                       anonymous_id=?, updated_at=?, last_login_at=?
                 WHERE id=?
                """,
                (unionid, session_key, nickname, avatar_url, anonymous_id, t, t, user_id)
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO wechat_users
                    (openid, unionid, session_key, nickname, avatar_url,
                     anonymous_id, created_at, updated_at, last_login_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (openid, unionid, session_key, nickname, avatar_url,
                 anonymous_id, t, t, t)
            )
            user_id = cur.lastrowid

        token = secrets.token_urlsafe(32)
        conn.execute(
            """
            INSERT INTO auth_sessions
                (token, user_id, created_at, expires_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token, user_id, t, t + 86400 * 30, t)
        )
    return {
        'user_id': user_id,
        'openid': openid,
        'nickname': nickname,
        'avatar_url': avatar_url,
        'session_token': token,
        'expires_at': t + 86400 * 30,
    }


def validate_session(token):
    if not token:
        return None
    t = _ts()
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT a.token, a.user_id, u.openid, u.nickname, u.avatar_url
              FROM auth_sessions a
              JOIN wechat_users u ON u.id = a.user_id
             WHERE a.token=? AND a.expires_at>?
            """,
            (token, t)
        ).fetchone()
        if not row:
            return None
        conn.execute(
            'UPDATE auth_sessions SET last_seen_at=? WHERE token=?',
            (t, token)
        )
        return dict(row)


def update_user_profile(user_id, nickname='', avatar_url=''):
    if not user_id:
        return None
    t = _ts()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE wechat_users
               SET nickname=COALESCE(NULLIF(?, ''), nickname),
                   avatar_url=COALESCE(NULLIF(?, ''), avatar_url),
                   updated_at=?
             WHERE id=?
            """,
            (nickname, avatar_url, t, user_id)
        )
        row = conn.execute(
            """
            SELECT id AS user_id, openid, nickname, avatar_url
              FROM wechat_users
             WHERE id=?
            """,
            (user_id,)
        ).fetchone()
        return dict(row) if row else None


def upsert_room_member(room_id, user_id, seat, player_name, is_host, socket_sid, status='online'):
    if not user_id or seat < 0:
        return
    t = _ts()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO room_members
                (room_id, user_id, seat, player_name, is_host, status,
                 socket_sid, created_at, updated_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(room_id, user_id) DO UPDATE SET
                seat=excluded.seat,
                player_name=excluded.player_name,
                is_host=excluded.is_host,
                status=excluded.status,
                socket_sid=excluded.socket_sid,
                updated_at=excluded.updated_at,
                last_seen_at=excluded.last_seen_at
            """,
            (room_id, user_id, seat, player_name, 1 if is_host else 0,
             status, socket_sid, t, t, t)
        )


def mark_member_offline(room_id, user_id, socket_sid=None):
    if not user_id:
        return
    t = _ts()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE room_members
               SET status='offline', socket_sid=?, updated_at=?, last_seen_at=?
             WHERE room_id=? AND user_id=?
            """,
            (socket_sid, t, t, room_id, user_id)
        )


def get_latest_member(user_id):
    if not user_id:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
              FROM room_members
             WHERE user_id=?
             ORDER BY updated_at DESC
             LIMIT 1
            """,
            (user_id,)
        ).fetchone()
        return dict(row) if row else None


def save_room_snapshot(room, round_result=None):
    if room is None:
        return
    room_state = room.get_room_state()
    game_states = None
    if room.current_game is not None:
        game_states = {}
        for p in room.players:
            if p is not None:
                game_states[str(p.seat)] = room.current_game.get_state(for_seat=p.seat)
    t = _ts()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO room_snapshots
                (room_id, room_state_json, game_states_json, round_result_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(room_id) DO UPDATE SET
                room_state_json=excluded.room_state_json,
                game_states_json=excluded.game_states_json,
                round_result_json=excluded.round_result_json,
                updated_at=excluded.updated_at
            """,
            (
                room.room_id,
                json.dumps(room_state, ensure_ascii=False),
                json.dumps(game_states, ensure_ascii=False) if game_states is not None else None,
                json.dumps(round_result, ensure_ascii=False) if round_result is not None else None,
                t,
            )
        )


def get_room_snapshot(room_id):
    with _connect() as conn:
        row = conn.execute(
            'SELECT * FROM room_snapshots WHERE room_id=?',
            (room_id,)
        ).fetchone()
        if not row:
            return None
        return {
            'room_id': row['room_id'],
            'room_state': json.loads(row['room_state_json']),
            'game_states': json.loads(row['game_states_json']) if row['game_states_json'] else None,
            'round_result': json.loads(row['round_result_json']) if row['round_result_json'] else None,
            'updated_at': row['updated_at'],
        }


def log_event(room_id, event, payload, user_id=None, seat=None):
    if not room_id:
        return
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO game_events (room_id, user_id, seat, event, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (room_id, user_id, seat, event,
             json.dumps(payload, ensure_ascii=False), _ts())
        )
