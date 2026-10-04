from pathlib import Path
import json
import time
import asyncio
from collections import deque
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"

LIVE_DATA_FILE = DATA_DIR / "live_sensor_data.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/sensor",
    tags=["sensor"]
)


# ============================================================
# DATA MODELS
# ============================================================

class SensorValues(BaseModel):

    ppg_ir: Optional[int] = None
    ppg_red: Optional[int] = None

    ecg: Optional[int] = None

    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None

    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None

    temperature_c: Optional[float] = None

    piezo: Optional[int] = None


class SensorPayload(BaseModel):

    device_id: str = Field(
        min_length=1,
        max_length=100
    )

    timestamp_ms: int

    sequence: int = Field(
        ge=0
    )

    sensors: SensorValues


# ============================================================
# RUNTIME STATE
# ============================================================

latest_data = None

last_sequence = None

packet_count = 0

packet_loss = 0

last_seen = None

history = deque(
    maxlen=300
)


# ============================================================
# WEBSOCKET CLIENTS
# ============================================================

websocket_clients = set()


# ============================================================
# JSON FILE INITIALIZATION
# ============================================================

def initialize_json_file():

    if LIVE_DATA_FILE.exists():
        return

    initial_data = {
        "status": {
            "connected": False,
            "device_id": None,
            "last_seen": None,
            "sequence": 0,
            "packet_count": 0,
            "packet_loss": 0
        },

        "sensors": {
            "ppg_ir": None,
            "ppg_red": None,
            "ecg": None,

            "accel_x": None,
            "accel_y": None,
            "accel_z": None,

            "gyro_x": None,
            "gyro_y": None,
            "gyro_z": None,

            "temperature_c": None,

            "piezo": None
        },

        "history": []
    }

    write_json(initial_data)


def write_json(data):

    temporary_file = LIVE_DATA_FILE.with_suffix(
        ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

    temporary_file.replace(
        LIVE_DATA_FILE
    )


def read_json():

    if not LIVE_DATA_FILE.exists():

        initialize_json_file()

    try:

        with open(
            LIVE_DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        initialize_json_file()

        with open(
            LIVE_DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)


initialize_json_file()


# ============================================================
# SAVE LIVE DATA
# ============================================================

def save_live_data(payload):

    global latest_data

    global last_sequence

    global packet_count

    global packet_loss

    global last_seen

    sequence = payload.sequence

    # --------------------------------------------------------
    # Detect missing packets
    # --------------------------------------------------------

    if last_sequence is not None:

        if sequence > last_sequence + 1:

            packet_loss += (
                sequence
                - last_sequence
                - 1
            )

    last_sequence = sequence

    packet_count += 1

    last_seen = time.time()

    # --------------------------------------------------------
    # Convert sensor payload
    # --------------------------------------------------------

    sensor_values = payload.sensors.model_dump()

    record = {
        "timestamp": time.time(),
        "timestamp_ms": payload.timestamp_ms,
        "device_id": payload.device_id,
        "sequence": sequence,
        "sensors": sensor_values
    }

    latest_data = record

    history.append(record)

    # --------------------------------------------------------
    # Determine connection status
    # --------------------------------------------------------

    connected = True

    # --------------------------------------------------------
    # Complete JSON document
    # --------------------------------------------------------

    document = {

        "status": {
            "connected": connected,
            "device_id": payload.device_id,
            "last_seen": last_seen,
            "sequence": sequence,
            "packet_count": packet_count,
            "packet_loss": packet_loss
        },

        "sensors": sensor_values,

        "latest": record,

        "history": list(history)
    }

    write_json(document)

    return document


# ============================================================
# RECEIVE SENSOR DATA
# ============================================================

@router.post("/data")
async def receive_sensor_data(
    payload: SensorPayload
):

    document = save_live_data(
        payload
    )

    # --------------------------------------------------------
    # Send to connected WebSocket clients
    # --------------------------------------------------------

    dead_clients = []

    for websocket in list(
        websocket_clients
    ):

        try:

            await websocket.send_json(
                document
            )

        except Exception:

            dead_clients.append(
                websocket
            )

    for websocket in dead_clients:

        websocket_clients.discard(
            websocket
        )

    return {
        "status": "ok",
        "sequence": payload.sequence,
        "file": str(LIVE_DATA_FILE)
    }


# ============================================================
# FRONTEND LIVE API
# ============================================================

@router.get("/live")
async def get_live_sensor_data():

    document = read_json()

    # --------------------------------------------------------
    # Dynamically calculate stale status
    # --------------------------------------------------------

    last_seen_value = (
        document
        .get("status", {})
        .get("last_seen")
    )

    if last_seen_value is None:

        document["status"]["connection_status"] = (
            "disconnected"
        )

    else:

        age = (
            time.time()
            - float(last_seen_value)
        )

        if age < 2:

            document["status"]["connection_status"] = (
                "connected"
            )

        elif age < 5:

            document["status"]["connection_status"] = (
                "stale"
            )

        else:

            document["status"]["connection_status"] = (
                "disconnected"
            )

    return document


# ============================================================
# SENSOR STATUS
# ============================================================

@router.get("/status")
async def get_sensor_status():

    document = read_json()

    status = document.get(
        "status",
        {}
    )

    last_seen_value = status.get(
        "last_seen"
    )

    if last_seen_value is None:

        connection_status = "disconnected"

        age = None

    else:

        age = (
            time.time()
            - float(last_seen_value)
        )

        if age < 2:

            connection_status = "connected"

        elif age < 5:

            connection_status = "stale"

        else:

            connection_status = "disconnected"

    return {
        "connection_status": connection_status,
        "last_seen_seconds": age,
        "device_id": status.get(
            "device_id"
        ),
        "sequence": status.get(
            "sequence",
            0
        ),
        "packet_count": status.get(
            "packet_count",
            0
        ),
        "packet_loss": status.get(
            "packet_loss",
            0
        )
    }


# ============================================================
# RAW JSON FILE CONTENT
# ============================================================

@router.get("/json")
async def get_json_data():

    return read_json()


# ============================================================
# WEBSOCKET
# ============================================================

@router.websocket("/ws")
async def sensor_websocket(
    websocket: WebSocket
):

    await websocket.accept()

    websocket_clients.add(
        websocket
    )

    try:

        # Immediately send current live JSON with evaluated connection status
        initial_doc = await get_live_sensor_data()
        await websocket.send_json(
            initial_doc
        )

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        websocket_clients.discard(
            websocket
        )

    except Exception:

        websocket_clients.discard(
            websocket
        )