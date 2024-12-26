import uuid
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

app = FastAPI()

security = HTTPBearer()

class SyncServer:
    # Mock method for authenticating a user via API key or password
    def authenticate_user(self) -> uuid.UUID:
        # Generate and return a default UUID for the admin
        return uuid.uuid4()

    # Mock method for mapping an API key to a user UUID
    def api_key_to_user(self, api_key: str) -> uuid.UUID:
        if api_key == "valid_api_key":
            return uuid.uuid4()
        raise Exception("Invalid API key")

def get_current_user(server: SyncServer, password: str, auth: HTTPAuthorizationCredentials = Depends(security)) -> uuid.UUID:
    try:
        api_key_or_password = auth.credentials
        if api_key_or_password == password:
            # User is admin, so return a default UUID
            return server.authenticate_user()
        # Try to match the API key to a user
        user_id = server.api_key_to_user(api_key_or_password)
        return user_id
    except Exception as e:
        # Catch and raise an HTTP error with a 403 code
        raise HTTPException(status_code=403, detail=f"Authentication error: {str(e)}")

@app.post("/authenticate")
async def authenticate(auth: HTTPAuthorizationCredentials = Depends(security)):
    server = SyncServer()
    password = "admin_password"  # The password for admin authentication

    try:
        # Get the current user (either admin or based on API key)
        user_id = get_current_user(server, password, auth)
        return {"message": f"Authenticated user ID: {user_id}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Authentication error: {str(e)}")

# Run the server using: uvicorn <filename>:app --reload
