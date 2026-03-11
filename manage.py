#!/usr/bin/env python
import os
import sys


def _is_runserver_command(argv: list[str]) -> bool:
    return any(str(arg).strip().lower() == "runserver" for arg in argv[1:])


def main():
    # Dev convenience: when running `python manage.py runserver`, execute Celery
    # tasks in-process so a separate worker/broker is not required.
    if _is_runserver_command(sys.argv):
        os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "1")
        os.environ.setdefault("CELERY_TASK_EAGER_PROPAGATES", "1")

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available on your "
            "PYTHONPATH environment variable? Did you forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
