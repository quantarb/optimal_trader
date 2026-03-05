from django.urls import path

from .views import labeling_config_form, labeling_symbol_detail


urlpatterns = [
    path("form/", labeling_config_form, name="labels-config-form"),
    path("symbol/<str:symbol>/", labeling_symbol_detail, name="labels-symbol-detail"),
]
