from django.urls import path

from .views import feature_preview_symbol

urlpatterns = [
    path("symbol/<str:symbol>/", feature_preview_symbol, name="features-symbol"),
]
