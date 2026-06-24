import importlib.util
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = 'django-insecure-change-me'

DEBUG = True

ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'fmp',
    'features',
    'labels',
    'ml',
    'pipeline',
    'trading',
]

if importlib.util.find_spec("django_celery_results") is not None:
    INSTALLED_APPS.append("django_celery_results")

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'wsgi.application'
ASGI_APPLICATION = 'asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            'timeout': 30,
        },
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "rpc://")
if "django_celery_results" in INSTALLED_APPS:
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "django-db")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_ALWAYS_EAGER = str(os.getenv("CELERY_TASK_ALWAYS_EAGER", "0")).strip().lower() in {"1", "true", "yes", "on"}
CELERY_TASK_EAGER_PROPAGATES = str(os.getenv("CELERY_TASK_EAGER_PROPAGATES", "1")).strip().lower() in {"1", "true", "yes", "on"}


# Quant Warehouse — ArcticDB store (code per env, data at QW_HOME).
QW_HOME = os.getenv("QW_HOME", str(Path.home() / ".quant-warehouse"))
QW_PRICE_PROVIDER = os.getenv("QW_PRICE_PROVIDER", "fmp")
QW_FUNDAMENTAL_PROVIDER = os.getenv("QW_FUNDAMENTAL_PROVIDER", "fmp")
QW_READ_PRICES = os.getenv("QW_READ_PRICES", "1")
QW_READ_FUNDAMENTALS = os.getenv("QW_READ_FUNDAMENTALS", "1")
QW_REFRESH_ENABLED = os.getenv("QW_REFRESH_ENABLED", "1")
QW_PROFILE_PROVIDER = os.getenv("QW_PROFILE_PROVIDER", "yfinance")
QW_READ_MACRO = os.getenv("QW_READ_MACRO", "1")
QW_MACRO_PROVIDER = os.getenv("QW_MACRO_PROVIDER", "fmp")
QW_SCREENER_ENABLED = os.getenv("QW_SCREENER_ENABLED", "1")
QW_SCREENER_PROVIDER = os.getenv("QW_SCREENER_PROVIDER", "fmp")

os.environ.setdefault("QW_HOME", QW_HOME)
os.environ.setdefault("QW_PRICE_PROVIDER", QW_PRICE_PROVIDER)
os.environ.setdefault("QW_FUNDAMENTAL_PROVIDER", QW_FUNDAMENTAL_PROVIDER)
os.environ.setdefault("QW_READ_PRICES", QW_READ_PRICES)
os.environ.setdefault("QW_READ_FUNDAMENTALS", QW_READ_FUNDAMENTALS)
os.environ.setdefault("QW_REFRESH_ENABLED", QW_REFRESH_ENABLED)
os.environ.setdefault("QW_PROFILE_PROVIDER", QW_PROFILE_PROVIDER)
os.environ.setdefault("QW_READ_MACRO", QW_READ_MACRO)
os.environ.setdefault("QW_MACRO_PROVIDER", QW_MACRO_PROVIDER)
os.environ.setdefault("QW_SCREENER_ENABLED", QW_SCREENER_ENABLED)
os.environ.setdefault("QW_SCREENER_PROVIDER", QW_SCREENER_PROVIDER)
