import asyncio
import websockets
import json

# 消息服务
class WebSocketClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.websocket = None

    # 连接
    async def connect(self):
        self.websocket = await websockets.connect(self.server_url)

    async def subscribe(self, topics):
        # 订阅感兴趣的主题
        await self.websocket.send(json.dumps({'subscribe': topics}))

    # 断开连接
    async def disconnect(self):
        # 添加延迟以确保disconnect有足够时间执行
        await asyncio.sleep(2)
        await self.websocket.close()

    # 发送消息
    async def send_message(self, topic, state):
        data = {'topic': topic, 'state': state}
        await self.websocket.send(json.dumps(data))

    # 接收消息
    async def receive_message(self):
        await self.websocket.send(json.dumps({'receive':''}))
        message = await self.websocket.recv()
        response = json.loads(message)
        print(f"Received response from server: {response}")

async def run():

    # 连接
    server_url = "ws://localhost:8765"
    client = WebSocketClient(server_url)

    try:
        await client.connect()
        print('connect')

        # 订阅
        topics = ['topic1'] 
        await client.subscribe(topics)

        # 发送消息
        await client.send_message('topic1', 'state1')
        await client.send_message('topic2', 'state2')

        # 接收消息
        await client.receive_message()

    finally:
        # 断开连接
        await client.disconnect()
        print('disconnect')

# 运行主事件循环
asyncio.run(run())
