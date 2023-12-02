import asyncio
import websockets
import json
import cv2

# 消息服务
class WebSocketClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.websocket = None

    # 连接
    async def connect(self):
        self.websocket = await websockets.connect(self.server_url)
        print('connect')

    # 订阅感兴趣的主题
    async def subscribe(self, topics):
        await self.websocket.send(json.dumps({'subscribe': topics}))

    # 取消订阅的主题
    async def unsubscribe(self, topics):
        await self.websocket.send(json.dumps({'unsubscribe': topics}))

    # 断开连接
    async def disconnect(self):
        # 添加延迟以确保disconnect有足够时间执行
        await asyncio.sleep(2)
        await self.websocket.close()
        print('disconnect')

    # 发送消息
    async def send_message(self, topic, state):
        data = {'topic': topic, 'state': state}
        try:
            await self.websocket.send(json.dumps(data))
        finally:
            pass

    # 接收消息
    async def receive_message(self):
        await self.websocket.send(json.dumps({'receive':''}))
        message = await self.websocket.recv()
        response = json.loads(message)
        return response