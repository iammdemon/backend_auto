
import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import backend.pdf_generator as pdf_generator
import backend.google_drive as google_drive
import zipfile
import tempfile

app = FastAPI()

# Allow all origins for simplicity, you can restrict this in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp"
OUTPUT_DIR = "output"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

async def process_and_generate_pdf(source_path, is_zip: bool):
    """Helper function to run processing and broadcast status."""
    loop = asyncio.get_running_loop()
    extracted_path = None
    try:
        await manager.broadcast("Starting processing...")

        if is_zip:
            extracted_path = tempfile.mkdtemp(dir=TEMP_DIR)
            await run_in_threadpool(unzip_file, source_path, extracted_path)
            process_root_folder = extracted_path
        else:
            process_root_folder = source_path
        
        await manager.broadcast("Generating PDF...")
        unique_filename = f"output_{uuid.uuid4()}.pdf"
        pdf_path = os.path.join(OUTPUT_DIR, unique_filename)
        await run_in_threadpool(pdf_generator.create_image_pdf, process_root_folder, pdf_path)
        
        await manager.broadcast(f"PDF_GENERATED:{pdf_path}")

    except Exception as e:
        await manager.broadcast(f"ERROR: {str(e)}")
    finally:
        # Clean up
        if is_zip and os.path.exists(source_path):
            os.remove(source_path) # Remove the uploaded zip file
        if extracted_path and os.path.exists(extracted_path):
            shutil.rmtree(extracted_path) # Remove the extracted folder
        elif not is_zip and os.path.exists(source_path):
            shutil.rmtree(source_path) # Remove the downloaded drive folder
        await manager.broadcast("Processing complete.")


@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    temp_file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.zip")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    asyncio.create_task(process_and_generate_pdf(temp_file_path, is_zip=True))
    return {"message": "Processing started. See status updates via WebSocket."}

@app.post("/process-drive/")
async def process_drive(drive_link: str = Form(...)):
    # This part is still blocking, but let's fix the core logic first.
    # A full solution would run this in a background task as well.
    downloaded_folder_path = await run_in_threadpool(google_drive.download_folder, drive_link)
    asyncio.create_task(process_and_generate_pdf(downloaded_folder_path, is_zip=False))
    return {"message": "Processing started. See status updates via WebSocket."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/download-pdf/")
async def download_pdf(path: str):
    print(f"Attempting to download file from path: {path}")
    if not os.path.exists(path):
        print(f"Error: File not found at path: {path}")
        # You might want to return a 404 error here instead of just printing
        return {"message": "File not found"}
    actual_filename = os.path.basename(path)
    print(f"Serving file: {actual_filename} from {path}")
    return FileResponse(path=path, filename=actual_filename, media_type='application/pdf')
