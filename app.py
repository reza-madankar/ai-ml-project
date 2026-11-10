import fastapi 
import asyncio


fapi = fastapi.FastAPI()


@fapi.get("/health-check")
def helath_check():
    return {"status": "Ok", "messgae":"API is running"}

@fapi.post("/sum-numbers")
def sum_numbers(numbers: list[int]):
    return {"status": "Ok", "message": sum(numbers) }
    
@fapi.websocket("/ws")
async def websocket_endpoint(websocket: fastapi.WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message received: {data}")
