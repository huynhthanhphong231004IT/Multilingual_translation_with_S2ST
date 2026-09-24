import pathlib
import asyncio
import numpy as np
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.websockets import WebSocketState
from fastapi.staticfiles import StaticFiles
from pipeline import Engine, Session

app = FastAPI()
ROOT = pathlib.Path(__file__).resolve().parent 
engine = Engine(ROOT)

static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
@app.get("/")
async def get_index():
    with open(ROOT / "client.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
    
@app.websocket("/ws/stream")
async def stream_audio(ws: WebSocket):
    await ws.accept()
    session = Session(engine)
    try:
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=0.5)
            except asyncio.TimeoutError:
                if ws.client_state == WebSocketState.CONNECTED:
                    flush_events = await session.force_flush()
                    for ev in flush_events:
                        if ev["type"] == "final":
                            audio_b64 = await engine.generate_tts(ev["translation"], ev["target_lang"])
                            ev["audio_b64"] = audio_b64
                            await ws.send_json(ev)
                continue

            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    action = data.get("action")
                    if action == "config":
                        session.update_config(data.get("src_lang", "vi"), data.get("target_lang", "en"))
                    elif action in ["finish", "stop"]:
                        flush_events = await session.force_flush()
                        for ev in flush_events:
                            if ev["type"] == "final" and ws.client_state == WebSocketState.CONNECTED:
                                audio_b64 = await engine.generate_tts(ev["translation"], ev["target_lang"])
                                ev["audio_b64"] = audio_b64
                                await ws.send_json(ev)
                        
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"type": "finished_all"})
                except Exception as e:
                    print("[WS CONFIG ERROR]", e)
            elif "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                samples_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32) / 32768.0
                events = await session.feed(samples_float32)
                for ev in events:
                    if ev["type"] == "final":
                        audio_b64 = await engine.generate_tts(ev["translation"], ev["target_lang"])
                        ev["audio_b64"] = audio_b64
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json(ev)
                await asyncio.sleep(0.001)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}")