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
        # numpy转换为list
        if type(state) == 'numpy.ndarray':
            data['state'] = state.tolist()
        await self.websocket.send(json.dumps(data))

    # 接收消息
    async def receive_message(self):
        await self.websocket.send(json.dumps({'receive':''}))
        message = await self.websocket.recv()
        response = json.loads(message)
        print(f"Received response from server: {response}")

async def main(url, topics):
    # 连接
    server_url = url
    client = WebSocketClient(server_url)
    try:
        # 连接
        await client.connect()
        # 订阅
        await client.subscribe(topics)

        # 取消订阅
        # topics = ['topic'] 
        # await client.unsubscribe(topics)
        
        # 主进程
        await process(client)

    finally:
        # 断开连接
        await client.disconnect()
    
async def process(client):
    cap = cv2.VideoCapture(0)
    try:
        while True:
            _, frame = cap.read()
            height, width, _ = frame.shape
            await client.send_message('camera_info', {'height': height, 'width': width})
            await client.receive_message()
    finally:
        cap.release()


url = "ws://localhost:8765"
topics = ['camera_info', 'topic']
# 运行主事件循环
asyncio.run(main(url, topics))
