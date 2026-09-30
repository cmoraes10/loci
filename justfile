test:
    python3 -m pytest -q

mcp:
    python3 adapters/mcp_server/server.py

check: test
    @echo "ok"
