
# Run

The application should run using the default flask implementation. This will create a local server that is single-threaded.

```bash
python app.py
```

For multiple instances of the application, the application should run using the gunicorn implementation. Requires to install gevent (`pip install gevent`).

```bash
gunicorn app:app -t 4 -b 0.0.0.0:5000 --timeout 5 -k gevent
```