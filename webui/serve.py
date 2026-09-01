from app import app


if __name__ == "__main__":
    print("Starting Kronos Web UI at http://localhost:7070")
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=7070)
