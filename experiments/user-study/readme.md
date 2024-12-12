
# Run

The application should run using the default flask implementation. This will create a local server that is single-threaded.

```bash
python app.py
```

For multiple instances of the application, the application should run using the gunicorn implementation.

```bash
gunicorn app:app -t 4 -b 0.0.0.0:5000
```