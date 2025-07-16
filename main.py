"""Entrypoint for running the FastAPI server."""

from src import server

if __name__ == "__main__":
    # Launch the FastAPI application
    server.run()
    