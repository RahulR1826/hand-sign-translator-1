import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)

rooms = {}

@sio.event
async def connect(sid, environ):
    print(f"{sid} connected")

@sio.event
async def disconnect(sid):
    print(f"{sid} disconnected")
    for room, members in rooms.items():
        if sid in members:
            members.remove(sid)
            await sio.emit("leave", {"id": sid}, room=room)
            break

@sio.event
async def join(sid, data):
    room = data['room']
    print(f"{sid} joined room {room}")
    rooms.setdefault(room, []).append(sid)
    await sio.save_session(sid, {"room": room})
    sio.enter_room(sid, room)
    await sio.emit("joined", {"id": sid}, room=room, skip_sid=sid)

@sio.event
async def signal(sid, data):
    await sio.emit("signal", data, to=data['to'])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
