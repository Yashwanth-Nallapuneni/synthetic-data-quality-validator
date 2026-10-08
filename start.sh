#!/usr/bin/env bash
exec uvicorn app.api:app --host 0.0.0.0 --port "${PORT:-10000}"
