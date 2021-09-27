"""
This script is based off the examples here:
https://blog.anvileight.com/posts/simple-python-http-server
"""
import os
import ssl
import argparse

from http.server import HTTPServer, SimpleHTTPRequestHandler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="server.py",
        description="""
    Host the Phantom documentation on a local HTTP server with SSL support.
    """,
    )

    parser.add_argument("keyfile", type=str)
    parser.add_argument("certfile", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--root_dir", default="_build/html", type=str)

    ARGS = parser.parse_args()

    os.chdir(ARGS.root_dir)

    httpd = HTTPServer(("localhost", ARGS.port), SimpleHTTPRequestHandler)
    httpd.socket = ssl.wrap_socket(
        httpd.socket, keyfile=ARGS.keyfile, certfile=ARGS.certfile, server_side=True
    )

    httpd.serve_forever()
