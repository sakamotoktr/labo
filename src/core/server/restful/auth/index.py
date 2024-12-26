from fastapi import FastAPI, HTTPException, APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional
import uuid

# Define logger (you can set up proper logging if needed)
import logging
logger = logging.getLogger(__name__)

# Simulating the SyncServer class
class SyncServer:
    def authenticate_user(self) -> uuid.UUID:
        # Simulate an admin authentication by returning a random UUID
        return uuid.uuid4()

    def api_key_to_user(self, api_key: str) -> uuid.UUID:
        if api_key == "valid_api_key":
            return uuid.uuid4()  # Simulate returning a user UUID
        else:
            raise Exception("Invalid API key")

# Define Pydantic models for the request and response
class AuthRequest(BaseModel):
    password: str = Field(..., description="Admin password provided when starting the server")

class AuthResponse(BaseModel):
    uuid: uuid.UUID = Field(..., description="UUID of the user")
    is_admin: Optional[bool] = Field(None, description="Whether the user is an admin")

# Setup the router
router = APIRouter()

def setup_auth_router(server: SyncServer, password: str) -> APIRouter:

    @router.post("/auth", tags=["auth"], response_model=AuthResponse)
    def authenticate_user(request: AuthRequest) -> AuthResponse:
        """
        Authenticates the user and sends response with User related data.
        This simulates user authentication.
        """
        # Initialize the is_admin flag
        is_admin = False

        # Clear interface (If there's an interface to clear, we would do it here)
        # In the Python version, we're not using the QueuingInterface, so no clearing required

        # Check if the request password matches the provided admin password
        if request.password != password:
            try:
                # Try to get the user by API key
                response_uuid = server.api_key_to_user(api_key=request.password)
            except Exception as e:
                raise HTTPException(status_code=403, detail=f"Authentication error: {str(e)}")
        else:
            # Admin password, so authenticate as admin
            is_admin = True
            response_uuid = server.authenticate_user()

        # Return the response with the UUID and is_admin flag
        return AuthResponse(uuid=response_uuid, is_admin=is_admin)

    return router

# Instantiate the FastAPI app
app = FastAPI()

# Setup a sample SyncServer instance and a password (you'd probably load this from settings)
server = SyncServer()
password = "admin_password"  # You would use a real password when starting the server

# Include the authentication router in the FastAPI app
app.include_router(setup_auth_router(server, password))

# Run the server using: uvicorn <filename>:app --reload
