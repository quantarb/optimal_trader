from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path("fmp/", include("fmp.urls")),
    path("features/", include("features.urls")),
    path("labels/", include("labels.urls")),
    path("ml/", include("ml.urls")),
]
