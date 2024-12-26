import importlib.util
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


def mount_static_files(app: FastAPI):
    static_files_path = os.path.join(os.path.dirname(importlib.util.find_spec("labo").origin), "server", "static_files")
    if os.path.exists(static_files_path):
        app.mount("/assets", StaticFiles(directory=os.path.join(static_files_path, "assets")), name="assets")

        @app.get("/labo_logo_transparent.png", include_in_schema=False)
        async def serve_spa():
            return FileResponse(os.path.join(static_files_path, "labo_logo_transparent.png"))

        common_paths = [
            "/", "/agents", "/data-sources", "/tools", "/agent-templates", "/human-templates",
            "/settings/profile", "/agents/{agent-id}/chat"
        ]
        for path in common_paths:
            @app.get(path, include_in_schema=False)
            async def serve_common_spa():
                return FileResponse(os.path.join(static_files_path, "index.html"))
