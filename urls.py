from django.contrib import admin
from django.urls import include, path
from fmp.views import form_tabs_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path("forms/", form_tabs_view, name="form-tabs"),
    path("fmp/", include("fmp.urls")),
    path("features/", include("features.urls")),
    path("labels/", include("labels.urls")),
    path("ml/", include("ml.urls")),
    path("pipeline/", include("pipeline.urls")),
    path("trading/", include("trading.urls")),
]
