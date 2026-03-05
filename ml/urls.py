from django.urls import path

from .views import model_artifact_detail_view, train_model_view

urlpatterns = [
    path("train/", train_model_view, name="train_model"),
    path("models/<int:artifact_id>/", model_artifact_detail_view, name="model_artifact_detail"),
]
