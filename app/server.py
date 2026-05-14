import http.server
import socketserver
import os
import json

PORT = 8000
METRICS_PATH = ".cache/metrics/metrics.json"

class MetricsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            content = "<h1>CureForge Agent - Real Time Metrics</h1>"
            if os.path.exists(METRICS_PATH):
                with open(METRICS_PATH, "r") as f:
                    data = json.load(f)
                content += f"<p><b>Last Updated:</b> {data.get('timestamp', 'N/A')}</p>"
                content += "<pre>" + json.dumps(data.get("metrics", {}), indent=2) + "</pre>"
            else:
                content += "<p>No metrics found. Run the agent first!</p>"
            
            self.wfile.write(content.encode())
        else:
            super().do_GET()

print(f"Starting server at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), MetricsHandler) as httpd:
    httpd.serve_forever()
