from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.sensor_receiver import router as sensor_router

app = FastAPI(
    title="Serenova Real-Time Sensor API",
    description="FastAPI ingestion and distribution service for Serenova Raspberry Pi Pico W live sensor telemetry.",
    version="1.0.0"
)

# Configure CORS for Vite frontend development and LAN testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Allow LAN origin access when testing from other devices
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sensor_router)


@app.get("/")
async def root():
    return {
        "service": "Serenova Sensor Ingestion Backend",
        "status": "operational",
        "endpoints": {
            "live_data": "/api/sensor/live",
            "status": "/api/sensor/status",
            "raw_json": "/api/sensor/json",
            "ingest_data": "POST /api/sensor/data",
            "websocket": "WS /api/sensor/ws",
            "docs": "/docs"
        }
    }