import socket
import numpy as np
import time
import json

# ===================== 配置项 =====================
# 服务端UDP地址（替换为你的Flask服务端IP，本地测试用127.0.0.1）
from .config import UDP_HOST, UDP_PORT

SERVER_UDP_IP = UDP_HOST
SERVER_UDP_PORT = UDP_PORT
SEND_FPS = 30  # 发送帧率（模拟实时动捕，建议30fps）


# ===================== 发送逻辑 =====================
class MocapStreamClient:
    def __init__(self, server_ip=SERVER_UDP_IP, server_port=SERVER_UDP_PORT):
        self.server_ip = server_ip
        self.server_port = server_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"✅ UDP客户端已创建，目标地址：{self.server_ip}:{self.server_port}")

    def send_frame(self, frame_data):
        """发送单帧动捕数据（frame_data为24x3的NumPy数组）"""
        try:
            frame_list = np.array(frame_data).tolist()
            send_data = json.dumps({"joints": frame_list})
            self.sock.sendto(
                send_data.encode("utf-8"), (self.server_ip, self.server_port)
            )
        except Exception as e:
            print(f"❌ 发送动捕帧失败：{str(e)}")
