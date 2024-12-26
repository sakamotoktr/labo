import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

# Setup for server and logger
interface: StreamingServerInterface = StreamingServerInterface
server = SyncServer(default_interface_factory=lambda: interface())
logger = get_logger(__name__)

# Password handling
password = None

def generate_password():
    import secrets
    return secrets.token_urlsafe(16)

random_password = os.getenv("LABO_SERVER_PASSWORD") or generate_password()

class CheckPasswordMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/v1/health/" or request.url.path == "/latest/health/":
            return await call_next(request)

        if request.headers.get("X-BARE-PASSWORD") == f"password {random_password}":
            return await call_next(request)

        return JSONResponse(
            content={"detail": "Unauthorized"},
            status_code=401,
        )

def generate_openapi_schema(app: FastAPI):
    # Generate OpenAPI schemas for different routes
    if not app.openapi_schema:
        app.openapi_schema = app.openapi()

    openai_docs, labo_docs = [app.openapi_schema.copy() for _ in range(2)]

    openai_docs["paths"] = {k: v for k, v in openai_docs["paths"].items() if k.startswith("/openai")}
    openai_docs["info"]["title"] = "OpenAI Assistants API"
    labo_docs["paths"] = {k: v for k, v in labo_docs["paths"].items() if not k.startswith("/openai")}
    labo_docs["info"]["title"] = "LABO API"
    labo_docs["components"]["schemas"]["LABOResponse"] = {
        "properties": LABOResponse.model_json_schema(ref_template="#/components/schemas/LABOResponse/properties/{model}")["$defs"]
    }

    for name, docs in [("openai", openai_docs), ("labo", labo_docs)]:
        if settings.cors_origins:
            docs["servers"] = [{"url": host} for host in settings.cors_origins]
        Path(f"openapi_{name}.json").write_text(json.dumps(docs, indent=2))

def create_application() -> FastAPI:
    print(f"\n[[ LABO server // v{__version__} ]]")

    if os.getenv("SENTRY_DSN"):
        import sentry_sdk
        sentry_sdk.init(
            dsn=os.getenv("SENTRY_DSN"),
            traces_sample_rate=1.0,
            _experiments={"continuous_profiling_auto_start": True},
        )

    debug_mode = "--debug" in sys.argv
    app = FastAPI(
        swagger_ui_parameters={"docExpansion": "none"},
        title="LABO",
        summary="Create LLM agents with long-term memory and custom tools 📚🦙",
        version="1.0.0",
        debug=debug_mode,
    )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        log.error(f"Unhandled error: {exc}", exc_info=True)
        if os.getenv("SENTRY_DSN"):
            sentry_sdk.capture_exception(exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred"},
        )

    # Exception handlers for various errors
    @app.exception_handler(NoResultFound)
    async def no_result_found_handler(request: Request, exc: NoResultFound):
        logger.error(f"NoResultFound: {exc}")
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ForeignKeyConstraintViolationError)
    async def foreign_key_constraint_handler(request: Request, exc: ForeignKeyConstraintViolationError):
        logger.error(f"ForeignKeyConstraintViolationError: {exc}")
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(UniqueConstraintViolationError)
    async def unique_key_constraint_handler(request: Request, exc: UniqueConstraintViolationError):
        logger.error(f"UniqueConstraintViolationError: {exc}")
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DatabaseTimeoutError)
    async def database_timeout_error_handler(request: Request, exc: DatabaseTimeoutError):
        logger.error(f"Timeout occurred: {exc}. Original exception: {exc.original_exception}")
        return JSONResponse(status_code=503, content={"detail": "The database is temporarily unavailable. Please try again later."})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(LABOAgentNotFoundError)
    async def agent_not_found_handler(request: Request, exc: LABOAgentNotFoundError):
        return JSONResponse(status_code=404, content={"detail": "Agent not found"})

    @app.exception_handler(LABOUserNotFoundError)
    async def user_not_found_handler(request: Request, exc: LABOUserNotFoundError):
        return JSONResponse(status_code=404, content={"detail": "User not found"})

    settings.cors_origins.append("https://app.labo.com")

    if os.getenv("LABO_SERVER_SECURE") == "true" or "--secure" in sys.argv:
        print(f"▶ Using secure mode with password: {random_password}")
        app.add_middleware(CheckPasswordMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include various routers
    for route in v1_routes:
        app.include_router(route, prefix=API_PREFIX)
        app.include_router(route, prefix="/latest", include_in_schema=False)

    # Admin routes
    app.include_router(users_router, prefix=ADMIN_PREFIX)
    app.include_router(organizations_router, prefix=ADMIN_PREFIX)

    # OpenAI routes
    app.include_router(openai_assistants_router, prefix=OPENAI_API_PREFIX)
    app.include_router(openai_chat_completions_router, prefix=OPENAI_API_PREFIX)

    # Auth routes
    app.include_router(setup_auth_router(server, interface, password), prefix=API_PREFIX)

    # Static files
    mount_static_files(app)

    @app.on_event("startup")
    def on_startup():
        generate_openapi_schema(app)

    @app.on_event("shutdown")
    def on_shutdown():
        global server

    return app

app = create_application()

def start_server(port: Optional[int] = None, host: Optional[str] = None, debug: bool = False):
    if debug:
        server_logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        server_logger.addHandler(stream_handler)

    if os.getenv("LOCAL_HTTPS") == "true" or "--localhttps" in sys.argv:
        uvicorn.run(
            app,
            host=host or "localhost",
            port=port or REST_DEFAULT_PORT,
            ssl_keyfile="certs/localhost-key.pem",
            ssl_certfile="certs/localhost.pem",
        )
        print(f"▶ Server running at: https://{host or 'localhost'}:{port or REST_DEFAULT_PORT}\n")
    else:
        uvicorn.run(
            app,
            host=host or "localhost",
            port=port or REST_DEFAULT_PORT,
        )
        print(f"▶ Server running at: http://{host or 'localhost'}:{port or REST_DEFAULT_PORT}\n")

    print(f"▶ View using ADE at: https://app.labo.com/development-servers/local/dashboard")

