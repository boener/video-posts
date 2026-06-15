import eventlet
eventlet.monkey_patch()

import os
import yaml
import sqlite3
import subprocess
import threading
import logging
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vertical-posts-dashboard'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

CONFIG_PATH = os.path.expanduser('~/vertical-posts/config.yaml')

DB_OPTIONS = {
    'karate':   '/mnt/storage/vertical-posts/data/blog_posts.db',
    'kombativ': '/mnt/storage/vertical-posts/data/blog_posts_kombativ.db',
}

pipeline_process = None
pipeline_thread = None
pipeline_lock = threading.Lock()
stop_requested = False
current_post_id = None


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    key_file = os.path.expanduser('~/OPENROUTER_API_KEY.txt')
    if os.path.exists(key_file):
        cfg.setdefault('api', {})['openrouter_api_key'] = open(key_file).read().strip()
    return cfg


def save_config(cfg):
    import copy
    out = copy.deepcopy(cfg)
    out.get('api', {}).pop('openrouter_api_key', None)
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(out, f, default_flow_style=False, allow_unicode=True)


def get_db():
    cfg = load_config()
    path = os.path.expanduser(cfg['paths']['db'])
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/status')
def status():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM blog_posts GROUP BY status")
    rows = cur.fetchall()
    conn.close()
    result = {'pending': 0, 'processing': 0, 'done': 0, 'failed': 0}
    for row in rows:
        result[row[0]] = row[1]
    return jsonify(result)


@app.route('/api/posts')
def posts():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT post_id, post_title, status, error_message, processed_at
        FROM blog_posts ORDER BY CAST(post_id AS INTEGER) ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route('/api/config')
def get_config():
    cfg = load_config()
    if 'api' in cfg and 'openrouter_api_key' in cfg['api']:
        cfg['api']['openrouter_api_key'] = '***'
    return jsonify(cfg)


@app.route('/api/config', methods=['POST'])
def post_config():
    data = request.get_json()
    cfg = load_config()

    field_map = {
        'models.script_writer': ('models', 'script_writer'),
        'models.tts': ('models', 'tts'),
        'models.image': ('models', 'image'),
        'voice.narrator_voice': ('voice', 'narrator_voice'),
        'voice.style_instructions': ('voice', 'style_instructions'),
        'images.style_prompt_prefix': ('images', 'style_prompt_prefix'),
        'paths.script_writer_prompt': ('paths', 'script_writer_prompt'),
        'ffmpeg.fps': ('ffmpeg', 'fps'),
        'ffmpeg.kenburns_scale': ('ffmpeg', 'kenburns_scale'),
        'rendering.mode': ('rendering', 'mode'),
        'rendering.remote_host': ('rendering', 'remote_host'),
        'rendering.remote_user': ('rendering', 'remote_user'),
        'rendering.remote_port': ('rendering', 'remote_port'),
        'rendering.remote_work_dir': ('rendering', 'remote_work_dir'),
        'script.target_duration_min_seconds': ('script', 'target_duration_min_seconds'),
        'script.target_duration_max_seconds': ('script', 'target_duration_max_seconds'),
        'script.target_segment_duration_seconds': ('script', 'target_segment_duration_seconds'),
    }

    for key, (section, field) in field_map.items():
        if key in data:
            if section not in cfg:
                cfg[section] = {}
            cfg[section][field] = data[key]

    save_config(cfg)
    return jsonify({'ok': True})


@app.route('/api/db-source')
def db_source():
    cfg = load_config()
    current = cfg['paths']['db']
    for name, path in DB_OPTIONS.items():
        if current == path:
            return jsonify({'source': name})
    return jsonify({'source': 'custom', 'path': current})


@app.route('/api/switch-db', methods=['POST'])
def switch_db():
    source = request.get_json().get('source')
    if source not in DB_OPTIONS:
        return jsonify({'error': 'invalid source'}), 400
    cfg = load_config()
    cfg['paths']['db'] = DB_OPTIONS[source]
    save_config(cfg)
    return jsonify({'ok': True})


@app.route('/api/retry', methods=['POST'])
def retry():
    """Reset a failed post to pending.

    mode='resume' — keep generated files on disk, pipeline will skip them.
    mode='reset'  — delete all generated files and start from scratch.
    """
    data = request.get_json()
    post_id = data['post_id']
    mode = data.get('mode', 'resume')

    if mode == 'reset':
        cfg = load_config()
        import shutil
        for subdir in ('audio', 'images'):
            d = os.path.join(os.path.expanduser(cfg['paths'][subdir]), str(post_id))
            if os.path.isdir(d):
                shutil.rmtree(d)
        for path_key, filename in (('scripts', f'{post_id}-script.json'),
                                   ('subtitles', f'{post_id}.ass'),
                                   ('videos', f'{post_id}.mp4')):
            p = os.path.join(os.path.expanduser(cfg['paths'][path_key]), filename)
            if os.path.exists(p):
                os.remove(p)

    conn = get_db()
    conn.execute(
        "UPDATE blog_posts SET status='pending', error_message=NULL, processed_at=NULL WHERE post_id=?",
        (post_id,)
    )
    conn.commit()
    conn.close()
    return jsonify({'ok': True})


@app.route('/api/run', methods=['POST'])
def run():
    global pipeline_thread, stop_requested, current_post_id
    with pipeline_lock:
        if pipeline_thread and pipeline_thread.is_alive():
            return jsonify({'error': 'already running'}), 409
        stop_requested = False
        current_post_id = None

    data = request.get_json()
    post_ids = data.get('post_ids', [])

    pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(post_ids,), daemon=True)
    pipeline_thread.start()
    return jsonify({'ok': True})


@app.route('/api/stop', methods=['POST'])
def stop():
    global stop_requested
    with pipeline_lock:
        stop_requested = True
        if pipeline_process:
            pipeline_process.kill()
    return jsonify({'ok': True})


@app.route('/api/pipeline-status')
def pipeline_status():
    with pipeline_lock:
        running = pipeline_thread is not None and pipeline_thread.is_alive()
        cid = current_post_id
    return jsonify({'running': running, 'current_post_id': cid})


def run_pipeline_thread(post_ids):
    global pipeline_process, stop_requested, current_post_id

    for post_id in post_ids:
        with pipeline_lock:
            if stop_requested:
                break
            current_post_id = post_id

        socketio.emit('log', {'line': f'--- Starting post {post_id} ---'})

        cmd = [
            os.path.expanduser('~/vertical-posts/venv/bin/python'),
            os.path.expanduser('~/vertical-posts/pipeline.py'),
            '--post-id', str(post_id),
            '--resume',
        ]

        with pipeline_lock:
            pipeline_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )

        for line in pipeline_process.stdout:
            socketio.emit('log', {'line': line.rstrip()})
            with pipeline_lock:
                if stop_requested:
                    pipeline_process.kill()
                    break

        pipeline_process.wait()

        with pipeline_lock:
            was_stopped = stop_requested
            stopped_post_id = current_post_id
            pipeline_process = None
            current_post_id = None

        if was_stopped and stopped_post_id is not None:
            try:
                conn = get_db()
                conn.execute(
                    "UPDATE blog_posts SET status='failed', error_message='Stopped by user' WHERE post_id=? AND status='processing'",
                    (stopped_post_id,)
                )
                conn.commit()
                conn.close()
            except Exception:
                pass
            socketio.emit('log', {'line': '--- Run stopped by user ---'})
            break

    socketio.emit('run_complete', {})


@socketio.on('connect')
def on_connect():
    try:
        cfg = load_config()
        log_path = os.path.expanduser(cfg['paths']['logs']) + '/pipeline.log'
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            last_lines = lines[-100:] if len(lines) > 100 else lines
            for line in last_lines:
                emit('log', {'line': line.rstrip()})
    except Exception:
        pass


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
