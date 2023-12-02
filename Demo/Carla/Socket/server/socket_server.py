import asyncio
import websockets
import json

# 用于存储每个主题的状态
topic_states = {}

# 用于存储每个连接
clients = set()

async def handle_client(websocket, path):
        # 将新连接添加到客户端集合
    clients.add(websocket)
    print(clients)
    subscriptions = []
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                # 订阅主题
                if 'subscribe' in data:
                    topics = data['subscribe']
                    for topic in topics:
                        subscriptions.append(topic)
                    print('订阅', subscriptions)

                # 取消订阅
                elif 'unsubscribe' in data:
                    topics = data['unsubscribe']
                    for topic in topics:
                        subscriptions.remove(topic)
                    print('取消订阅', subscriptions)

                # 客户端发送消息
                elif 'topic' in data and 'state' in data:
                    topic = data['topic']
                    state = data['state']
                    # 更新主题下的状态
                    topic_states.setdefault(topic, {})
                    topic_states[topic] = state

                # 客户端接收消息
                elif 'receive' in data:
                    # 只接收订阅的消息
                    message = {}
                    for i in subscriptions:
                        if i in topic_states:
                            message[i] = topic_states[i]
                    message_json = json.dumps(message)
                    await websocket.send(message_json)
 
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    except websockets.ConnectionClosed:
        # 删除连接
        print("Client connection closed.")
        clients.remove(websocket)

    finally:
        print('finally')

# 启动 WebSocket 服务器
start_server = websockets.serve(handle_client, "localhost", 8765)

# 运行事件循环
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
