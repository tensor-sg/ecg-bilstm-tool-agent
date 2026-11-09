"""Tools API: the agent's tools as plain REST endpoints, usable without an LLM.

GET  /tools                -> catalogue with JSON schemas (exactly what the LLM is offered)
GET  /tools/{name}         -> one tool's schema
POST /tools/{name}/invoke  -> run a tool with JSON arguments
"""

import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.deps import AppState, get_state
from app.schemas import ToolInfo, ToolInvokeRequest, ToolInvokeResponse
from app.tools import ToolArgumentError, ToolResourceNotFound, ToolUnavailable, UnknownToolError

router = APIRouter(prefix="/tools", tags=["tools"])


def _info(schema: dict) -> ToolInfo:
    f = schema["function"]
    return ToolInfo(name=f["name"], description=f["description"], parameters=f["parameters"])


@router.get("", response_model=list[ToolInfo])
async def list_tools(state: AppState = Depends(get_state)):
    return [_info(t) for t in state.agent.tools.schemas]


@router.get("/{name}", response_model=ToolInfo)
async def get_tool(name: str, state: AppState = Depends(get_state)):
    try:
        return _info(state.agent.tools.schema(name))
    except UnknownToolError:
        raise HTTPException(404, f"Unknown tool '{name}'") from None


@router.post("/{name}/invoke", response_model=ToolInvokeResponse)
async def invoke_tool(name: str, req: ToolInvokeRequest, state: AppState = Depends(get_state)):
    t = time.perf_counter()
    try:
        result = await run_in_threadpool(state.agent.tools.invoke, name, req.arguments)
    except UnknownToolError:
        raise HTTPException(404, f"Unknown tool '{name}'") from None
    except ToolArgumentError as exc:
        raise HTTPException(422, f"Invalid arguments for {name}: {exc}") from None
    except ToolResourceNotFound as exc:
        raise HTTPException(404, {"message": str(exc), **exc.details}) from None
    except ToolUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return ToolInvokeResponse(tool=name, arguments=req.arguments, result=result,
                              duration_ms=round((time.perf_counter() - t) * 1000, 1))
