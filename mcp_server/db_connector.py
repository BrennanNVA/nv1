"""MCP server for TimescaleDB database access from Cursor chat."""

import asyncio
import logging
import sys
from pathlib import Path

import asyncpg
from mcp.server import Server
from mcp.types import Resource, TextContent, Tool

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nova.core.config import load_config
from src.nova.core.validation import SQLQueryValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("nova-aetus-db")

# Global config and pool
config = None
pool = None


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available database resources."""
    return [
        Resource(
            uri="timescaledb://tables",
            name="Database Tables",
            description="List of all tables in TimescaleDB",
            mimeType="text/plain",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """
    Read resource content.

    Args:
        uri: Resource URI

    Returns:
        Resource content as string
    """
    if uri == "timescaledb://tables":
        if not pool:
            await connect_db()

        async with pool.acquire() as conn:
            tables = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            )

            table_list = "\n".join([row["table_name"] for row in tables])
            return f"Database Tables:\n{table_list}"

    return f"Unknown resource: {uri}"


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available database tools."""
    return [
        Tool(
            name="execute_query",
            description="Execute a SELECT query on TimescaleDB. Only SELECT queries are allowed for safety.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_tables",
            description="List all tables in the database",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="describe_table",
            description="Get schema information for a table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to describe",
                    },
                },
                "required": ["table_name"],
            },
        ),
        Tool(
            name="get_portfolio_summary",
            description="Get summary of open positions and P&L",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_recent_signals",
            description="Get latest generated trading signals",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of signals to return",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="get_system_health",
            description="Get latest system metrics and health status",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle tool calls.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        List of text content results
    """
    if not pool:
        await connect_db()

    if name == "execute_query":
        query = arguments.get("query", "")

        # Validate query for safety
        try:
            query = SQLQueryValidator.validate_select_only(query)
        except ValueError as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error: {e}",
                )
            ]

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)

                if not rows:
                    return [
                        TextContent(
                            type="text",
                            text="Query executed successfully. No rows returned.",
                        )
                    ]

                # Format results
                result_text = "Query Results:\n\n"
                result_text += " | ".join(rows[0].keys()) + "\n"
                result_text += "-" * 80 + "\n"

                for row in rows[:100]:  # Limit to 100 rows
                    result_text += " | ".join(str(val) for val in row.values()) + "\n"

                if len(rows) > 100:
                    result_text += f"\n... ({len(rows) - 100} more rows)"

                return [TextContent(type="text", text=result_text)]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error executing query: {e}",
                )
            ]

    elif name == "list_tables":
        try:
            async with pool.acquire() as conn:
                tables = await conn.fetch(
                    """
                    SELECT table_name,
                           (SELECT COUNT(*) FROM information_schema.columns
                            WHERE table_name = t.table_name) as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """
                )

                result_text = "Database Tables:\n\n"
                for row in tables:
                    result_text += f"- {row['table_name']} ({row['column_count']} columns)\n"

                return [TextContent(type="text", text=result_text)]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error listing tables: {e}",
                )
            ]

    elif name == "describe_table":
        table_name = arguments.get("table_name", "")

        # Validate table name
        try:
            table_name = SQLQueryValidator.validate_table_name(table_name)
        except ValueError as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error: {e}",
                )
            ]

        try:
            async with pool.acquire() as conn:
                columns = await conn.fetch(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = $1
                    ORDER BY ordinal_position
                """,
                    table_name,
                )

                if not columns:
                    return [
                        TextContent(
                            type="text",
                            text=f"Table '{table_name}' not found.",
                        )
                    ]

                result_text = f"Table: {table_name}\n\n"
                result_text += "Column Name | Data Type | Nullable\n"
                result_text += "-" * 50 + "\n"

                for col in columns:
                    result_text += (
                        f"{col['column_name']} | {col['data_type']} | {col['is_nullable']}\n"
                    )

                return [TextContent(type="text", text=result_text)]

        except Exception as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error describing table: {e}",
                )
            ]

    elif name == "get_portfolio_summary":
        try:
            async with pool.acquire() as conn:
                positions = await conn.fetch(
                    """
                    SELECT symbol, quantity, entry_price, current_price,
                           pnl, pnl_pct, status, entry_time
                    FROM portfolio_positions
                    WHERE status = 'OPEN'
                    ORDER BY entry_time DESC
                """
                )

                if not positions:
                    return [TextContent(type="text", text="No open positions found.")]

                total_pnl = sum(p["pnl"] or 0 for p in positions)

                result_text = f"Portfolio Summary (Total Unrealized P&L: ${total_pnl:.2f})\n\n"
                result_text += "Symbol | Qty | Entry | Current | P&L | P&L % | Opened\n"
                result_text += "-" * 80 + "\n"

                for p in positions:
                    result_text += f"{p['symbol']} | {p['quantity']} | ${p['entry_price']:.2f} | "
                    result_text += f"${p['current_price'] or 0:.2f} | ${p['pnl'] or 0:.2f} | "
                    result_text += (
                        f"{p['pnl_pct'] or 0:.2f}% | {p['entry_time'].strftime('%Y-%m-%d %H:%M')}\n"
                    )

                return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting portfolio summary: {e}")]

    elif name == "get_recent_signals":
        limit = arguments.get("limit", 10)
        try:
            async with pool.acquire() as conn:
                signals = await conn.fetch(
                    """
                    SELECT timestamp, symbol, signal_type, direction, strength, confluence_score
                    FROM trading_signals
                    ORDER BY timestamp DESC
                    LIMIT $1
                """,
                    limit,
                )

                if not signals:
                    return [TextContent(type="text", text="No signals found.")]

                result_text = f"Latest {len(signals)} Trading Signals:\n\n"
                result_text += "Time | Symbol | Type | Direction | Strength | Confluence\n"
                result_text += "-" * 80 + "\n"

                for s in signals:
                    result_text += f"{s['timestamp'].strftime('%m-%d %H:%M')} | {s['symbol']} | "
                    result_text += f"{s['signal_type']} | {s['direction']} | {s['strength']:.2f} | "
                    result_text += f"{s['confluence_score']:.2f}\n"

                return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting recent signals: {e}")]

    elif name == "get_system_health":
        try:
            async with pool.acquire() as conn:
                metrics = await conn.fetch(
                    """
                    SELECT DISTINCT ON (metric_name) timestamp, metric_name, metric_value, labels
                    FROM system_metrics
                    WHERE timestamp > NOW() - INTERVAL '5 minutes'
                    ORDER BY metric_name, timestamp DESC
                """
                )

                if not metrics:
                    return [TextContent(type="text", text="No recent system metrics found.")]

                result_text = "Latest System Health Metrics (last 5 min):\n\n"
                result_text += "Metric | Value | Labels | Time\n"
                result_text += "-" * 80 + "\n"

                for m in metrics:
                    labels = m["labels"] if m["labels"] else ""
                    result_text += f"{m['metric_name']} | {m['metric_value']:.4f} | {labels} | {m['timestamp'].strftime('%H:%M:%S')}\n"

                return [TextContent(type="text", text=result_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error getting system health: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def connect_db() -> None:
    """Connect to TimescaleDB."""
    global config, pool

    try:
        config = load_config()
        pool = await asyncpg.create_pool(
            host=config.data.timescale_host,
            port=config.data.timescale_port,
            user=config.data.timescale_user,
            password=config.data.timescale_password,
            database=config.data.timescale_db,
            min_size=1,
            max_size=5,
        )
        logger.info("Connected to TimescaleDB for MCP server")
    except Exception as e:
        logger.error(f"Failed to connect to TimescaleDB: {e}")
        raise


async def main() -> None:
    """Run MCP server."""
    await connect_db()

    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
