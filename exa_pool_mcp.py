# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]>=1.0.0",
#     "httpx>=0.27.0",
# ]
# ///
"""
Exa Pool MCP Server
Wraps the Exa Pool API as an MCP server for Claude Code.

This server provides access to Exa's AI-powered search
capabilities through a API pool.
"""

from mcp.server.fastmcp import FastMCP
import httpx
import json
import logging
import os
from typing import Optional, List

# Configure logging (critical for stdio transport - never use print!)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("exa-pool")

# API Configuration
EXA_POOL_BASE_URL = os.getenv("EXA_POOL_BASE_URL", "")
EXA_POOL_API_KEY = os.getenv("EXA_POOL_API_KEY", "")

# HTTP client timeout settings
TIMEOUT = httpx.Timeout(30.0, connect=5.0)


# ============================================================================
# Helper Functions
# ============================================================================

def format_error(status_code: int, message: str) -> str:
    """Format error messages consistently."""
    return f"Error {status_code}: {message}"


def format_json_response(data: dict) -> str:
    """Format JSON response for readability."""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to format JSON: {e}")
        return str(data)


async def make_exa_request(
    endpoint: str,
    method: str = "POST",
    data: Optional[dict] = None,
    params: Optional[dict] = None
) -> str:
    """
    Make a request to the Exa Pool API with proper error handling.

    Args:
        endpoint: API endpoint (e.g., "/search", "/contents")
        method: HTTP method (default: POST)
        data: Request body data
        params: Query parameters

    Returns:
        JSON response as formatted string or error message
    """
    url = f"{EXA_POOL_BASE_URL}{endpoint}"
    headers = {
        "x-api-key": EXA_POOL_API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "exa-pool-mcp-server/1.0"
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if method == "POST":
                response = await client.post(url, json=data, headers=headers)
            elif method == "GET":
                response = await client.get(url, params=params, headers=headers)
            else:
                return f"Error: Unsupported HTTP method: {method}"

            # Handle specific HTTP status codes
            if response.status_code == 401:
                logger.error("Authentication failed - API key may be invalid")
                return format_error(401, "Authentication failed. API key may be invalid.")

            if response.status_code == 403:
                logger.error("Access forbidden")
                return format_error(403, "Access denied.")

            if response.status_code == 404:
                logger.error(f"Endpoint not found: {endpoint}")
                return format_error(404, f"Endpoint not found: {endpoint}")

            if response.status_code == 429:
                logger.warning("Rate limited")
                return format_error(429, "Rate limited. Please try again later.")

            if response.status_code >= 500:
                logger.error(f"Server error: {response.status_code}")
                return format_error(
                    response.status_code,
                    "Exa Pool server error. The service may be temporarily unavailable."
                )

            # Raise for other 4xx/5xx errors
            response.raise_for_status()

            # Parse and return JSON response
            result = response.json()
            return format_json_response(result)

    except httpx.TimeoutException:
        logger.error(f"Request timeout for {endpoint}")
        return "Error: Request timed out after 30 seconds. The Exa Pool API may be slow or unavailable."

    except httpx.ConnectError as e:
        logger.error(f"Connection error: {e}")
        return f"Error: Unable to connect to Exa Pool API at {EXA_POOL_BASE_URL}. Please check the service status."

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e}")
        return format_error(
            e.response.status_code,
            f"HTTP request failed: {e.response.reason_phrase}"
        )

    except ValueError as e:
        logger.error(f"Invalid JSON response: {e}")
        return "Error: Received invalid JSON response from Exa Pool API."

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return f"Error: {type(e).__name__}: {str(e)}"


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
async def exa_search(
    query: str,
    num_results: int = 10,
    search_type: str = "auto",
    include_text: bool = False
) -> str:
    """
    Search the web using Exa's AI-powered search engine.

    This tool performs intelligent web searches using neural embeddings and
    returns relevant results with metadata.

    Args:
        query: The search query (required)
        num_results: Number of results to return (1-100, default: 10)
        search_type: Search algorithm - "auto" (default), "neural", "fast", or "deep"
        include_text: Whether to include page text content in results (default: False)

    Returns:
        JSON-formatted search results containing URLs, titles, and metadata

    Example:
        exa_search("latest developments in AI agents", num_results=5)
    """
    # Validate inputs
    if not query or not query.strip():
        return "Error: query parameter is required and cannot be empty"

    if not 1 <= num_results <= 100:
        return "Error: num_results must be between 1 and 100"

    if search_type not in ["auto", "neural", "fast", "deep"]:
        return "Error: search_type must be one of: auto, neural, fast, deep"

    # Build request payload
    payload = {
        "query": query.strip(),
        "numResults": num_results,
        "type": search_type
    }

    # Add text content if requested
    if include_text:
        payload["contents"] = {"text": True}

    logger.info(f"Searching Exa: query='{query}', num_results={num_results}, type={search_type}")

    return await make_exa_request("/search", data=payload)


@mcp.tool()
async def exa_get_contents(
    urls: List[str],
    include_text: bool = True,
    include_html: bool = False
) -> str:
    """
    Get clean, parsed content from one or more web pages.

    Retrieves full page contents with automatic caching and live crawling fallback.

    Args:
        urls: List of URLs to fetch content from (required, max 100)
        include_text: Include cleaned text content (default: True)
        include_html: Include raw HTML content (default: False)

    Returns:
        JSON-formatted page contents with text, HTML, and metadata

    Example:
        exa_get_contents(["https://example.com/article"])
    """
    # Validate inputs
    if not urls or len(urls) == 0:
        return "Error: urls parameter is required and cannot be empty"

    if len(urls) > 100:
        return "Error: Maximum 100 URLs allowed per request"

    # Validate URL format (basic check)
    for url in urls:
        if not url.startswith(("http://", "https://")):
            return f"Error: Invalid URL format: {url}. URLs must start with http:// or https://"

    # Build request payload
    payload = {
        "urls": urls,
        "text": include_text
    }

    if include_html:
        payload["htmlContent"] = True

    logger.info(f"Fetching contents for {len(urls)} URL(s)")

    return await make_exa_request("/contents", data=payload)


@mcp.tool()
async def exa_find_similar(
    url: str,
    num_results: int = 10,
    include_text: bool = False
) -> str:
    """
    Find web pages similar to a given URL using semantic similarity.

    Uses Exa's embeddings to find pages with similar meaning and content.

    Args:
        url: The reference URL to find similar pages for (required)
        num_results: Number of similar pages to return (1-100, default: 10)
        include_text: Whether to include page text content (default: False)

    Returns:
        JSON-formatted list of similar pages with URLs, titles, and metadata

    Example:
        exa_find_similar("https://arxiv.org/abs/2307.06435", num_results=5)
    """
    # Validate inputs
    if not url or not url.strip():
        return "Error: url parameter is required and cannot be empty"

    if not url.startswith(("http://", "https://")):
        return f"Error: Invalid URL format. URLs must start with http:// or https://"

    if not 1 <= num_results <= 100:
        return "Error: num_results must be between 1 and 100"

    # Build request payload
    payload = {
        "url": url.strip(),
        "numResults": num_results
    }

    if include_text:
        payload["contents"] = {"text": True}

    logger.info(f"Finding similar pages to: {url}")

    return await make_exa_request("/findSimilar", data=payload)


@mcp.tool()
async def exa_answer(
    query: str,
    include_text: bool = False
) -> str:
    """
    Get an AI-generated answer to a question using Exa's Answer API.

    Performs a search and uses an LLM to generate a direct answer with citations.

    Args:
        query: The question to answer (required)
        include_text: Whether to include full text from cited sources (default: False)

    Returns:
        JSON-formatted answer with citations and source metadata

    Example:
        exa_answer("What is the latest valuation of SpaceX?")
    """
    # Validate inputs
    if not query or not query.strip():
        return "Error: query parameter is required and cannot be empty"

    # Build request payload
    payload = {
        "query": query.strip(),
        "text": include_text
    }

    logger.info(f"Getting answer for: {query}")

    return await make_exa_request("/answer", data=payload)


@mcp.tool()
async def exa_create_research(
    instructions: str,
    model: str = "exa-research"
) -> str:
    """
    Create an asynchronous deep research task.

    Initiates an in-depth web research task that explores the web, gathers sources,
    and returns structured results with citations. This is an async operation.

    Args:
        instructions: Research task description (required, max 4096 characters)
        model: Research model - "exa-research-fast", "exa-research" (default), or "exa-research-pro"

    Returns:
        JSON with researchId for tracking the task status

    Example:
        exa_create_research("Summarize the latest papers on vision transformers")
    """
    # Validate inputs
    if not instructions or not instructions.strip():
        return "Error: instructions parameter is required and cannot be empty"

    if len(instructions) > 4096:
        return "Error: instructions must be 4096 characters or less"

    if model not in ["exa-research-fast", "exa-research", "exa-research-pro"]:
        return "Error: model must be one of: exa-research-fast, exa-research, exa-research-pro"

    # Build request payload
    payload = {
        "instructions": instructions.strip(),
        "model": model
    }

    logger.info(f"Creating research task with model: {model}")

    return await make_exa_request("/research/v1", data=payload)


@mcp.tool()
async def exa_get_research(research_id: str) -> str:
    """
    Get the status and results of a research task.

    Retrieves the current status and results (if completed) of a previously
    created research task.

    Args:
        research_id: The research task ID returned from exa_create_research (required)

    Returns:
        JSON with task status and results (if completed)

    Example:
        exa_get_research("research_abc123")
    """
    # Validate inputs
    if not research_id or not research_id.strip():
        return "Error: research_id parameter is required and cannot be empty"

    logger.info(f"Getting research task: {research_id}")

    return await make_exa_request(
        f"/research/v1/{research_id.strip()}",
        method="GET"
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the MCP server using stdio transport."""
    logger.info("Starting Exa Pool MCP Server")
    if EXA_POOL_BASE_URL:
        logger.info(f"API configured: {EXA_POOL_BASE_URL[:20]}...")
    else:
        logger.warning("EXA_BASE_URL not set - server may not function correctly")
    logger.info("Transport: stdio")

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
