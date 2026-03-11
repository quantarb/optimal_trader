from django.urls import path

from .views import feature_job_rows_json, feature_preview_form, feature_preview_symbol, feature_symbol_table_json

urlpatterns = [
    path("form/", feature_preview_form, name="features-form"),
    path("jobs/recent/", feature_job_rows_json, name="features-jobs-recent"),
    path("symbols/table/", feature_symbol_table_json, name="features-symbol-table"),
    path("symbol/<int:feature_run_id>/<str:symbol>/", feature_preview_symbol, name="features-symbol-run"),
    path("symbol/<str:symbol>/", feature_preview_symbol, name="features-symbol"),
]
