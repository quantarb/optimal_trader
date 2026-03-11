from django.urls import path

from .views import (
    model_artifact_detail_view,
    model_artifact_symbol_predictions_view,
)

urlpatterns = [
    path("models/<int:artifact_id>/", model_artifact_detail_view, name="model_artifact_detail"),
    path("models/<int:artifact_id>/symbol/<str:symbol>/", model_artifact_symbol_predictions_view, name="model_artifact_symbol_predictions"),
]
