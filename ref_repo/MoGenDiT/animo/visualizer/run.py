from flask import Flask, jsonify, request  # 新增request
from flask_socketio import SocketIO, emit
import os
from pathlib import Path
import socket
import json
import threading
import time
import numpy as np
import eventlet  # 显式导入eventlet

eventlet.monkey_patch()  # 关键：修补标准库，确保eventlet正常工作

# 说明：需要先安装Flask等依赖
# pip install flask-socketio==5.3.6 python-socketio==5.8.0 python-engineio==4.9.0 eventlet==0.33.3

# ========== 配置项 ==========
from .config import APP_HOST, APP_PORT, UDP_HOST, UDP_PORT

APP_HOST = APP_HOST
APP_PORT = APP_PORT
UDP_HOST = UDP_HOST
UDP_PORT = UDP_PORT

# ========== 全局状态 + 线程锁（核心修复：线程安全） ==========
MOCAP_STATE = {
    "is_listening": False,
    "thread": None,
    "udp_socket": None,
    "latest_joints": [],
}
# 新增：全局锁，保护MOCAP_STATE的读写
state_lock = threading.Lock()

# ========== 初始化Flask + WebSocket（核心修复：心跳配置） ==========
app = Flask(__name__)
# 核心修复：SocketIO配置（心跳保活+强制eventlet）
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",  # 强制使用eventlet引擎
    ping_interval=5,  # 每5秒发一次心跳（ping）
    ping_timeout=15,  # 15秒未响应则判定断开（默认60秒，缩短更灵敏）
    max_http_buffer_size=1024 * 1024,  # 增大缓冲区，避免大数据包溢出
)


# ========== 工具函数：线程安全的状态更新（核心修复） ==========
def update_mocap_state(key, value):
    """线程安全地更新MOCAP_STATE"""
    with state_lock:
        MOCAP_STATE[key] = value


def get_mocap_state(key):
    """线程安全地读取MOCAP_STATE"""
    with state_lock:
        return MOCAP_STATE.get(key, None)


# ========== 核心：动捕数据监听线程函数（修复跨线程推送） ==========
def mocap_listen_thread():
    """独立线程: 监听UDP动捕数据, 直到is_listening为False"""
    udp_socket = None
    try:
        # 创建UDP Socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind((UDP_HOST, UDP_PORT))
        udp_socket.settimeout(1.0)
        update_mocap_state("udp_socket", udp_socket)
        print(f"✅ 动捕监听已启动: UDP {UDP_HOST}:{UDP_PORT}")

        # 循环监听
        while get_mocap_state("is_listening"):
            try:
                data, addr = udp_socket.recvfrom(4096)
                # print(f"📥 收到UDP数据来自: {addr} (长度: {len(data)}字节)")

                # 解析数据
                mocap_data = json.loads(data.decode("utf-8"))
                joints = mocap_data.get("joints", [])

                # 线程安全更新最新数据
                update_mocap_state("latest_joints", joints)

                # 核心修复：通过socketio.start_background_task在SocketIO上下文推送
                socketio.start_background_task(
                    lambda: socketio.emit("mocap_update", {"joints": joints})
                )

            except socket.timeout:
                continue  # 超时仅检查停止信号
            except json.JSONDecodeError:
                print(f"❌ 动捕数据格式错误: 非JSON格式")
            except Exception as e:
                print(f"❌ 接收UDP数据出错: {str(e)}")

    except Exception as e:
        print(f"❌ 监听线程异常: {str(e)}")
    finally:
        # 强制关闭UDP Socket（修复僵尸线程）
        if udp_socket:
            try:
                udp_socket.close()
            except:
                pass
        # 清理状态
        update_mocap_state("udp_socket", None)
        update_mocap_state("thread", None)
        update_mocap_state("is_listening", False)
        print("🛑 动捕监听线程已安全退出")


# ========== 接口1：启动动捕监听 ==========
@app.route("/mocap/start", methods=["GET"])
def start_mocap():
    with state_lock:
        if MOCAP_STATE["is_listening"]:
            return jsonify(
                {
                    "code": 200,
                    "msg": "动捕监听已在运行中",
                    "data": {"is_listening": True},
                }
            )

        # 设置启动状态
        MOCAP_STATE["is_listening"] = True
        # 启动监听线程（daemon=True：主进程退出时自动销毁）
        listen_thread = threading.Thread(target=mocap_listen_thread, daemon=True)
        MOCAP_STATE["thread"] = listen_thread
        listen_thread.start()

    return jsonify(
        {"code": 200, "msg": "动捕监听启动成功", "data": {"is_listening": True}}
    )


# ========== 接口2：停止动捕监听（修复退出逻辑） ==========
@app.route("/mocap/stop", methods=["GET"])
def stop_mocap():
    with state_lock:
        if not MOCAP_STATE["is_listening"]:
            return jsonify(
                {"code": 200, "msg": "动捕监听未运行", "data": {"is_listening": False}}
            )

        # 1. 设置停止信号
        MOCAP_STATE["is_listening"] = False
        # 2. 强制关闭UDP Socket（触发线程退出）
        udp_socket = MOCAP_STATE.get("udp_socket")
        if udp_socket:
            try:
                udp_socket.close()
            except:
                pass
        # 3. 等待线程退出（超时强制）
        thread = MOCAP_STATE.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
            if thread.is_alive():
                print("⚠️ 监听线程未正常退出，已强制清理")

    return jsonify(
        {"code": 200, "msg": "动捕监听停止成功", "data": {"is_listening": False}}
    )


# ========== 接口3：获取当前监听状态 + 最新关节数据 ==========
@app.route("/mocap/status", methods=["GET"])
def get_mocap_status():
    with state_lock:
        status_data = {
            "is_listening": MOCAP_STATE["is_listening"],
            "latest_joints": MOCAP_STATE["latest_joints"],
        }
    return jsonify({"code": 200, "data": status_data})


# ========== 前端页面路由 ==========
@app.route("/")
def index():
    current_folder = Path(__file__).resolve().parent
    html_path = current_folder / "templates" / "index_vis.html"
    if not html_path.exists():
        return jsonify({"code": 404, "msg": "前端页面未找到，请检查index_vis.html路径"})
    return open(html_path, "r", encoding="utf-8").read()


# ========== WebSocket事件（保留+优化日志） ==========
@socketio.on("connect")
def handle_websocket_connect():
    print(
        f"\n🔗 [WebSocket] 客户端已连接 (ID: {request.sid})"
    )  # request.sid是客户端唯一标识
    emit("connect_success", {"msg": "WebSocket连接成功"})


@socketio.on("request_latest_data")
def send_latest_data():
    with state_lock:
        latest_joints = MOCAP_STATE["latest_joints"]
    emit("mocap_update", {"joints": latest_joints})
    print(f"📤 响应客户端请求，推送最新动捕数据（{len(latest_joints)}个关节）")


@socketio.on("disconnect")
def handle_websocket_disconnect():
    print(f"\n🔌 [WebSocket] 客户端已断开连接 (ID: {request.sid})")


# ========== 主函数（修复SocketIO引擎） ==========
if __name__ == "__main__":
    print(f"\n🚀 启动Animo动作可视化系统...")
    print(
        f"🌐 前端页面访问地址：http://{APP_HOST if APP_HOST != '0.0.0.0' else 'localhost'}:{APP_PORT}"
    )
    print(f"📡 接口文档：")
    print(f"  - 启动监听：GET http://localhost:{APP_PORT}/mocap/start")
    print(f"  - 停止监听：GET http://localhost:{APP_PORT}/mocap/stop")
    print(f"  - 查询状态：GET http://localhost:{APP_PORT}/mocap/status")
    print("=" * 50)
    # 核心修复：显式指定async_mode，禁用debug（避免重复启动）
    socketio.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        debug=False,
        use_reloader=False,  # 禁用自动重载（多线程下会冲突）
    )
