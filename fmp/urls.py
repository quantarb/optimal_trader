from django.urls import path

from .views import (
    economic_indicators_form,
    macro_series_detail,
    symbol_chart,
    symbol_detail,
    treasury_rates_form,
    universe_screener,
    universe_screener_download_start,
    universe_screener_download_status,
    universe_screener_form,
)

urlpatterns = [
    path("economic-indicators/form/", economic_indicators_form, name="economic-indicators-form"),
    path("treasury-rates/form/", treasury_rates_form, name="treasury-rates-form"),
    path("macro/series/<str:code>/", macro_series_detail, name="macro-series-detail"),
    path("universe-screener/form/", universe_screener_form, name="universe-screener-form"),
    path("universe-screener/", universe_screener, name="universe-screener"),
    path("universe-screener/download/start/", universe_screener_download_start, name="universe-screener-download-start"),
    path("universe-screener/download/status/<str:job_id>/", universe_screener_download_status, name="universe-screener-download-status"),
    path("symbol/<str:symbol>/", symbol_detail, name="symbol-detail"),
    path("symbol/<str:symbol>/chart/", symbol_chart, name="symbol-chart"),
]
