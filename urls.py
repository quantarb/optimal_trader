from django.urls import include, path

urlpatterns = [
    path("", include("trading.urls")),
    path("trading/", include("trading.urls")),
]
