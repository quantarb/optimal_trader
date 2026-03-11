from __future__ import annotations

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

try:
    from celery import Celery
except Exception:  # pragma: no cover - optional dependency in local dev
    app = None
else:
    app = Celery("optimal_trader")
    app.config_from_object("django.conf:settings", namespace="CELERY")
    app.autodiscover_tasks()
