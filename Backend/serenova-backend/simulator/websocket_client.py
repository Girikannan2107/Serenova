import asyncio
import json
import websockets

async def test_websocket():
    uri = "ws://127.0.0.1:8000/api/sensor/ws"
    print(f"[*] Connecting to WebSocket at {uri}...")
    async with websockets.connect(uri) as websocket:
        print("[+] WebSocket connected successfully!")
        
        # Receive initial state snapshot
        initial_msg = await websocket.recv()
        data = json.loads(initial_msg)
        print("[+] Received initial snapshot:")
        print("    Status:", data.get("status"))
        print("    Sensors:", data.get("sensors"))
        print("    History length:", len(data.get("history", [])))
        return True

if __name__ == "__main__":
    asyncio.run(test_websocket())
