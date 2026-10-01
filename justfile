test:
    python3 -m pytest -q

mcp:
    python3 -m loci.adapters.mcp_server.server

check: test
    @echo "ok"
